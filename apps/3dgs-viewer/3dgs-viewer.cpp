#include "polymer-core/lib-polymer.hpp"
#include "polymer-app-base/glfw-app.hpp"
#include "polymer-app-base/camera-controllers.hpp"
#include "polymer-app-base/wrappers/gl-imgui.hpp"
#include "polymer-gfx-gl/gl-api.hpp"
#include "polymer-gfx-gl/gl-texture-view.hpp"
#include "polymer-model-io/gaussian-splat-io.hpp"

#include "stb/stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

using namespace polymer;

/////////////////////////////////
//   Small Utilities           //
/////////////////////////////////

inline uint16_t float_to_half(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mantissa = x & 0x007FFFFFu;
    const int32_t exponent = int32_t((x >> 23) & 0xFFu) - 127 + 15;
    if (exponent >= 31) return uint16_t(sign | 0x7C00u);
    if (exponent <= 0)
    {
        if (exponent < -10) return uint16_t(sign);
        mantissa |= 0x00800000u;
        const uint32_t shift = uint32_t(14 - exponent);
        uint16_t h = uint16_t(mantissa >> shift);
        if ((mantissa >> (shift - 1)) & 1u) h++;
        return uint16_t(sign | h);
    }
    uint16_t h = uint16_t(sign | (uint32_t(exponent) << 10) | (mantissa >> 13));
    if (mantissa & 0x1000u) h++; // round-to-nearest, carry rolls into the exponent correctly
    return h;
}

// GPU pass timing via timestamp query pairs, ring-buffered so results are read
// four frames later without ever stalling the pipeline.
enum gs_pass : uint32_t
{
    gs_pass_cov3d = 0,
    gs_pass_preprocess,
    gs_pass_scan,
    gs_pass_sort,
    gs_pass_boundary,
    gs_pass_render,
    gs_pass_total,
    gs_pass_count
};

inline const char * gs_pass_name(uint32_t p)
{
    switch (p)
    {
        case gs_pass_cov3d: return "cov3d";
        case gs_pass_preprocess: return "preprocess";
        case gs_pass_scan: return "scan+expand";
        case gs_pass_sort: return "radix sort";
        case gs_pass_boundary: return "boundary";
        case gs_pass_render: return "render";
        case gs_pass_total: return "gpu total";
        default: return "?";
    }
}

struct gpu_pass_timers
{
    static constexpr uint32_t ring = 4;

    GLuint queries[gs_pass_count][ring][2] {};
    bool used[gs_pass_count][ring] {};
    float last_ms[gs_pass_count] {};
    uint64_t frame = 0;

    void initialize() { glGenQueries(gs_pass_count * ring * 2, &queries[0][0][0]); }
    uint32_t slot() const { return uint32_t(frame % ring); }

    void poll_oldest()
    {
        const uint32_t s = slot();
        for (uint32_t p = 0; p < gs_pass_count; ++p)
        {
            if (!used[p][s]) continue;
            GLuint64 t0 = 0, t1 = 0;
            glGetQueryObjectui64v(queries[p][s][0], GL_QUERY_RESULT, &t0);
            glGetQueryObjectui64v(queries[p][s][1], GL_QUERY_RESULT, &t1);
            last_ms[p] = float(double(t1 - t0) * 1e-6);
            used[p][s] = false;
        }
    }

    void begin(uint32_t p) { glQueryCounter(queries[p][slot()][0], GL_TIMESTAMP); }
    void end(uint32_t p) { glQueryCounter(queries[p][slot()][1], GL_TIMESTAMP); used[p][slot()] = true; }
    void next_frame() { frame++; }
};

/////////////////////////////////
//   Gaussian Splat Renderer   //
/////////////////////////////////

// Frame-constant parameters shared by every pass. Must mirror the std140 FrameParams
// block in assets/shaders/3dgs/*.comp exactly (208 bytes).
struct gs_frame_params
{
    float4x4 view_3dgs;       // flipZ * view_rh * model: +z forward, y up, x right
    float4x4 viewproj;        // proj_gl * view_rh * model
    float4 cam_pos_model;     // camera position in model space (for SH view directions)
    float4 bg_color;
    uint32_t resolution[2];
    uint32_t tile_grid[2];
    uint32_t tile_bits;
    uint32_t num_gaussians;
    uint32_t sh_degree;
    float scale_modifier;
    float tan_fovx;
    float tan_fovy;
    float focal_x;
    float focal_y;
};
static_assert(sizeof(gs_frame_params) == 208, "gs_frame_params must match the std140 FrameParams block");

struct gs_render_settings
{
    int sh_degree = 3;
    float scale_modifier = 1.0f;
    float3 bg_color {0.0f, 0.0f, 0.0f};
    bool flip_model = true; // COLMAP-trained scenes are y-down; 180-degree X rotation (not a mirror)
};

class gaussian_splat_renderer
{

public:

    gaussian_splat_renderer() = default;
    ~gaussian_splat_renderer() = default;

    void initialize(uint32_t width, uint32_t height)
    {
        load_shaders();

        gpu_state_buffer_.set_buffer_data(4 * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        indirect_buffer_.set_buffer_data(3 * 4 * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        scan_len_counts_.set_buffer_data(sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        scan_len_hist_.set_buffer_data(sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        frame_ubo_.set_buffer_data(sizeof(gs_frame_params), nullptr, GL_DYNAMIC_DRAW);

        glCreateBuffers(1, &stats_buffer_);
        glNamedBufferStorage(stats_buffer_, 3 * 16, nullptr, GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
        stats_ptr_ = glMapNamedBufferRange(stats_buffer_, 0, 3 * 16, GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

        timers_.initialize();
        resize(width, height);
    }

    void resize(uint32_t width, uint32_t height)
    {
        if (width_ == width && height_ == height) return;
        width_ = width;
        height_ = height;
        tiles_x_ = (width_ + 15) / 16;
        tiles_y_ = (height_ + 15) / 16;

        const uint32_t num_tiles = tiles_x_ * tiles_y_;
        tile_bits_ = 1;
        while ((1u << tile_bits_) < num_tiles) tile_bits_++;

        output_texture_ = gl_texture_2d();
        output_texture_.setup(width_, height_, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        tile_boundary_buffer_.set_buffer_data(num_tiles * 2 * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
    }

    void set_scene(const gaussian_splat_scene & scene)
    {
        num_gaussians_ = static_cast<uint32_t>(scene.vertices.size());
        scene_sh_degree_ = scene.sh_degree;
        if (num_gaussians_ == 0) return;

        // Repack into GPU-friendly layouts: hot position+opacity (16B), rotation+scale for
        // the cov3d pass (32B), and SH coefficients as packed half floats (96B max).
        std::vector<float4> pos_opacity(num_gaussians_);
        std::vector<float4> rot_scale(size_t(num_gaussians_) * 2);
        std::vector<uint32_t> sh_packed(size_t(num_gaussians_) * 24, 0);

        for (uint32_t i = 0; i < num_gaussians_; ++i)
        {
            const gaussian_vertex & v = scene.vertices[i];
            pos_opacity[i] = float4(v.position.x, v.position.y, v.position.z, v.scale_opacity.w);
            rot_scale[size_t(i) * 2 + 0] = v.rotation; // stored (w, x, y, z), matching the shader
            rot_scale[size_t(i) * 2 + 1] = float4(v.scale_opacity.x, v.scale_opacity.y, v.scale_opacity.z, 0.0f);
            for (uint32_t k = 0; k < 24; ++k)
            {
                const uint32_t lo = float_to_half(v.shs[k * 2 + 0]);
                const uint32_t hi = float_to_half(v.shs[k * 2 + 1]);
                sh_packed[size_t(i) * 24 + k] = lo | (hi << 16);
            }
        }

        pos_opacity_buffer_.set_buffer_data(pos_opacity.size() * sizeof(float4), pos_opacity.data(), GL_STATIC_DRAW);
        rot_scale_buffer_.set_buffer_data(rot_scale.size() * sizeof(float4), rot_scale.data(), GL_STATIC_DRAW);
        sh_buffer_.set_buffer_data(sh_packed.size() * sizeof(uint32_t), sh_packed.data(), GL_STATIC_DRAW);

        cov3d_buffer_.set_buffer_data(size_t(num_gaussians_) * 6 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        tile_counts_buffer_.set_buffer_data(size_t(num_gaussians_) * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        offsets_buffer_.set_buffer_data(size_t(num_gaussians_) * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        attr_render_buffer_.set_buffer_data(size_t(num_gaussians_) * 32, nullptr, GL_DYNAMIC_DRAW);
        attr_emit_buffer_.set_buffer_data(size_t(num_gaussians_) * 16, nullptr, GL_DYNAMIC_DRAW);

        const uint32_t n = num_gaussians_;
        scan_len_counts_.set_buffer_sub_data(sizeof(uint32_t), 0, &n);

        instance_capacity_ = 0;
        allocate_instance_buffers(std::max(1000000u, num_gaussians_ * 4u));

        cov3d_dirty_ = true;
        stats_instances_ = 0;
        stats_total_unclamped_ = 0;
    }

    void render(const perspective_camera & cam, const gs_render_settings & s)
    {
        if (num_gaussians_ == 0)
        {
            const float clear_color[4] = { s.bg_color.x, s.bg_color.y, s.bg_color.z, 1.0f };
            glClearTexImage(output_texture_, 0, GL_RGBA, GL_FLOAT, clear_color);
            return;
        }

        const std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        timers_.poll_oldest();
        timers_.begin(gs_pass_total);

        update_frame_params(cam, s);

        if (cov3d_dirty_ || s.scale_modifier != last_scale_modifier_)
        {
            cov3d_dirty_ = false;
            last_scale_modifier_ = s.scale_modifier;
            timers_.begin(gs_pass_cov3d);
            precomp_cov3d_shader_->bind_ubo(0, frame_ubo_);
            precomp_cov3d_shader_->bind_ssbo(1, rot_scale_buffer_);
            precomp_cov3d_shader_->bind_ssbo(2, cov3d_buffer_);
            precomp_cov3d_shader_->dispatch_and_barrier((num_gaussians_ + 255) / 256, 1, 1, GL_SHADER_STORAGE_BARRIER_BIT);
            timers_.end(gs_pass_cov3d);
        }

        timers_.begin(gs_pass_preprocess);
        preprocess_shader_->bind_ubo(0, frame_ubo_);
        preprocess_shader_->bind_ssbo(1, pos_opacity_buffer_);
        preprocess_shader_->bind_ssbo(2, cov3d_buffer_);
        preprocess_shader_->bind_ssbo(3, sh_buffer_);
        preprocess_shader_->bind_ssbo(4, tile_counts_buffer_);
        preprocess_shader_->bind_ssbo(5, attr_render_buffer_);
        preprocess_shader_->bind_ssbo(6, attr_emit_buffer_);
        preprocess_shader_->dispatch_and_barrier((num_gaussians_ + 255) / 256, 1, 1, GL_SHADER_STORAGE_BARRIER_BIT);
        timers_.end(gs_pass_preprocess);

        timers_.begin(gs_pass_scan);
        run_scan(scan_len_counts_, tile_counts_buffer_, offsets_buffer_, (num_gaussians_ + 2047) / 2048, false);

        prepare_shader_->bind_ubo(0, frame_ubo_);
        prepare_shader_->bind_ssbo(1, gpu_state_buffer_);
        prepare_shader_->bind_ssbo(2, tile_counts_buffer_);
        prepare_shader_->bind_ssbo(3, offsets_buffer_);
        prepare_shader_->bind_ssbo(4, indirect_buffer_);
        prepare_shader_->bind_ssbo(5, scan_len_hist_);
        prepare_shader_->dispatch_and_barrier(1, 1, 1, GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

        expand_shader_->bind_ubo(0, frame_ubo_);
        expand_shader_->bind_ssbo(1, gpu_state_buffer_);
        expand_shader_->bind_ssbo(2, tile_counts_buffer_);
        expand_shader_->bind_ssbo(3, offsets_buffer_);
        expand_shader_->bind_ssbo(4, attr_emit_buffer_);
        expand_shader_->bind_ssbo(5, keys_buffer_[0]);
        expand_shader_->bind_ssbo(6, payloads_buffer_[0]);
        expand_shader_->dispatch_and_barrier((num_gaussians_ + 255) / 256, 1, 1, GL_SHADER_STORAGE_BARRIER_BIT);
        timers_.end(gs_pass_scan);

        timers_.begin(gs_pass_sort);
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, indirect_buffer_);
        run_radix_sort(0, 0, true);
        timers_.end(gs_pass_sort);

        timers_.begin(gs_pass_boundary);
        glClearNamedBufferData(tile_boundary_buffer_, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        tile_boundary_shader_->bind_ubo(0, frame_ubo_);
        tile_boundary_shader_->bind_ssbo(1, gpu_state_buffer_);
        tile_boundary_shader_->bind_ssbo(2, keys_buffer_[0]);
        tile_boundary_shader_->bind_ssbo(3, tile_boundary_buffer_);
        tile_boundary_shader_->bind();
        glDispatchComputeIndirect(2 * 16);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        timers_.end(gs_pass_boundary);

        timers_.begin(gs_pass_render);
        render_shader_->bind_ubo(0, frame_ubo_);
        render_shader_->bind_ssbo(1, attr_render_buffer_);
        render_shader_->bind_ssbo(2, tile_boundary_buffer_);
        render_shader_->bind_ssbo(3, payloads_buffer_[0]);
        render_shader_->bind_image(4, output_texture_, GL_WRITE_ONLY, GL_RGBA8);
        render_shader_->dispatch_and_barrier(tiles_x_, tiles_y_, 1,
            GL_TEXTURE_FETCH_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        timers_.end(gs_pass_render);

        timers_.end(gs_pass_total);
        timers_.next_frame();

        update_async_stats();

        last_frame_time_ms_ = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    }

    // Runs the radix sort on synthetic data and verifies against std::stable_sort.
    // pattern: 0 = random, 1 = already sorted, 2 = reverse sorted, 3 = all equal
    bool run_sort_test(uint32_t count, int pattern)
    {
        allocate_instance_buffers(count);

        std::vector<uint32_t> keys(count), payloads(count);
        uint64_t state = 0x853c49e6748fea9bULL;
        for (uint32_t i = 0; i < count; ++i)
        {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            if (pattern == 0) keys[i] = uint32_t(state >> 32);
            else if (pattern == 1) keys[i] = i / 3u;
            else if (pattern == 2) keys[i] = count - i;
            else keys[i] = 0xCAFEBABEu;
            payloads[i] = i;
        }

        keys_buffer_[0].set_buffer_sub_data(count * sizeof(uint32_t), 0, keys.data());
        payloads_buffer_[0].set_buffer_sub_data(count * sizeof(uint32_t), 0, payloads.data());

        const uint32_t wgs = (count + 8191) / 8192;
        const uint32_t gpu_state[4] = { count, count, wgs, instance_capacity_ };
        gpu_state_buffer_.set_buffer_sub_data(sizeof(gpu_state), 0, gpu_state);
        const uint32_t hist_len = 256 * wgs;
        scan_len_hist_.set_buffer_sub_data(sizeof(uint32_t), 0, &hist_len);

        run_radix_sort(wgs, (hist_len + 2047) / 2048, false);

        std::vector<uint32_t> sorted_keys(count), sorted_payloads(count);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glGetNamedBufferSubData(keys_buffer_[0], 0, count * sizeof(uint32_t), sorted_keys.data());
        glGetNamedBufferSubData(payloads_buffer_[0], 0, count * sizeof(uint32_t), sorted_payloads.data());

        std::vector<std::pair<uint32_t, uint32_t>> reference(count);
        for (uint32_t i = 0; i < count; ++i) reference[i] = { keys[i], payloads[i] };
        std::stable_sort(reference.begin(), reference.end(), [](const std::pair<uint32_t, uint32_t> & a, const std::pair<uint32_t, uint32_t> & b) { return a.first < b.first; });

        for (uint32_t i = 0; i < count; ++i)
        {
            if (sorted_keys[i] != reference[i].first || sorted_payloads[i] != reference[i].second)
            {
                printf("    sort mismatch at %u: got (%08x, %u) expected (%08x, %u)\n",
                    i, sorted_keys[i], sorted_payloads[i], reference[i].first, reference[i].second);
                return false;
            }
        }
        return true;
    }

    GLuint get_output_texture() const { return output_texture_; }
    uint32_t get_instance_count() const { return stats_instances_; } // ~2 frames stale
    uint32_t get_unclamped_total() const { return stats_total_unclamped_; }
    float get_frame_time_ms() const { return last_frame_time_ms_; }
    const float * get_pass_ms() const { return timers_.last_ms; }

private:

    void load_shaders()
    {
        precomp_cov3d_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/precomp_cov3d.comp"));
        preprocess_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/preprocess.comp"));
        scan_block_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/scan_block.comp"));
        scan_sums_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/scan_sums.comp"));
        scan_apply_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/scan_apply.comp"));
        prepare_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/prepare_dispatch.comp"));
        expand_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/expand_keys.comp"));
        radix_hist_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/radix_hist.comp"));
        radix_scatter_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/radix_scatter.comp"));
        tile_boundary_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/tile_boundary.comp"));
        render_shader_ = std::make_unique<gl_shader_compute>(read_file_text("../assets/shaders/3dgs/render.comp"));

        hist_shift_loc_ = glGetUniformLocation(radix_hist_shader_->handle(), "u_shift");
        scatter_shift_loc_ = glGetUniformLocation(radix_scatter_shader_->handle(), "u_shift");
    }

    void allocate_instance_buffers(uint32_t needed_capacity)
    {
        constexpr uint32_t max_capacity = 48000000; // ~768 MB of key/payload buffers; beyond this stay clamped
        needed_capacity = std::min(needed_capacity, max_capacity);
        if (needed_capacity <= instance_capacity_) return;
        instance_capacity_ = std::min(std::max(needed_capacity, instance_capacity_ + instance_capacity_ / 2), max_capacity);

        for (int i = 0; i < 2; ++i)
        {
            keys_buffer_[i].set_buffer_data(size_t(instance_capacity_) * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
            payloads_buffer_[i].set_buffer_data(size_t(instance_capacity_) * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);
        }

        const uint32_t max_sort_wgs = (instance_capacity_ + 8191) / 8192;
        hist_buffer_.set_buffer_data(size_t(max_sort_wgs) * 256 * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);

        const uint32_t max_scan_len = std::max(num_gaussians_, max_sort_wgs * 256);
        scan_block_sums_buffer_.set_buffer_data((size_t(max_scan_len) / 2048 + 2) * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);

        const uint32_t capacity = instance_capacity_;
        gpu_state_buffer_.set_buffer_sub_data(sizeof(uint32_t), 3 * sizeof(uint32_t), &capacity);

        std::cout << "instance buffers: capacity " << instance_capacity_ << " ("
                  << (size_t(instance_capacity_) * 16) / (1024 * 1024) << " MB)" << std::endl;
    }

    void update_frame_params(const perspective_camera & cam, const gs_render_settings & s)
    {
        const float aspect = float(width_) / float(height_);
        const float4x4 model = s.flip_model
            ? float4x4{{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}}  // 180-degree X rotation
            : float4x4{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        const float4x4 flip_z {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};

        const float4x4 view_model = cam.get_view_matrix() * model;
        const float3 p = cam.pose.position;

        gs_frame_params params {};
        params.view_3dgs = flip_z * view_model;
        params.viewproj = cam.get_projection_matrix(aspect) * view_model;
        params.cam_pos_model = s.flip_model ? float4(p.x, -p.y, -p.z, 1.0f) : float4(p, 1.0f); // model inverse is itself
        params.bg_color = float4(s.bg_color, 1.0f);
        params.resolution[0] = width_;
        params.resolution[1] = height_;
        params.tile_grid[0] = tiles_x_;
        params.tile_grid[1] = tiles_y_;
        params.tile_bits = tile_bits_;
        params.num_gaussians = num_gaussians_;
        params.sh_degree = std::min(uint32_t(s.sh_degree), scene_sh_degree_);
        params.scale_modifier = s.scale_modifier;
        params.tan_fovy = std::tan(cam.vfov * 0.5f);
        params.tan_fovx = params.tan_fovy * aspect;
        params.focal_x = float(width_) / (2.0f * params.tan_fovx);
        params.focal_y = float(height_) / (2.0f * params.tan_fovy);

        frame_ubo_.set_buffer_sub_data(sizeof(params), 0, &params);
    }

    // Three-dispatch hierarchical exclusive scan. When indirect, pass A/C consume the
    // dispatch args at indirect slot 1 (written by prepare_dispatch on the GPU).
    void run_scan(GLuint len_buffer, GLuint in_buffer, GLuint out_buffer, uint32_t groups, bool indirect)
    {
        scan_block_shader_->bind_ssbo(0, len_buffer);
        scan_block_shader_->bind_ssbo(1, in_buffer);
        scan_block_shader_->bind_ssbo(2, out_buffer);
        scan_block_shader_->bind_ssbo(3, scan_block_sums_buffer_);
        scan_block_shader_->bind();
        if (indirect) glDispatchComputeIndirect(1 * 16);
        else glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        scan_sums_shader_->bind_ssbo(0, len_buffer);
        scan_sums_shader_->bind_ssbo(3, scan_block_sums_buffer_);
        scan_sums_shader_->dispatch_and_barrier(1, 1, 1, GL_SHADER_STORAGE_BARRIER_BIT);

        scan_apply_shader_->bind_ssbo(0, len_buffer);
        scan_apply_shader_->bind_ssbo(2, out_buffer);
        scan_apply_shader_->bind_ssbo(3, scan_block_sums_buffer_);
        scan_apply_shader_->bind();
        if (indirect) glDispatchComputeIndirect(1 * 16);
        else glDispatchCompute(groups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Four 8-bit LSD passes over 32-bit keys; even pass count returns data to buffer 0.
    void run_radix_sort(uint32_t direct_wgs, uint32_t direct_scan_groups, bool indirect)
    {
        for (uint32_t pass = 0; pass < 4; ++pass)
        {
            const uint32_t src = pass & 1u;
            const uint32_t dst = src ^ 1u;

            glProgramUniform1ui(radix_hist_shader_->handle(), hist_shift_loc_, pass * 8);
            radix_hist_shader_->bind_ssbo(0, gpu_state_buffer_);
            radix_hist_shader_->bind_ssbo(1, keys_buffer_[src]);
            radix_hist_shader_->bind_ssbo(2, hist_buffer_);
            radix_hist_shader_->bind();
            if (indirect) glDispatchComputeIndirect(0);
            else glDispatchCompute(direct_wgs, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            run_scan(scan_len_hist_, hist_buffer_, hist_buffer_, direct_scan_groups, indirect);

            glProgramUniform1ui(radix_scatter_shader_->handle(), scatter_shift_loc_, pass * 8);
            radix_scatter_shader_->bind_ssbo(0, gpu_state_buffer_);
            radix_scatter_shader_->bind_ssbo(1, keys_buffer_[src]);
            radix_scatter_shader_->bind_ssbo(2, payloads_buffer_[src]);
            radix_scatter_shader_->bind_ssbo(3, keys_buffer_[dst]);
            radix_scatter_shader_->bind_ssbo(4, payloads_buffer_[dst]);
            radix_scatter_shader_->bind_ssbo(5, hist_buffer_);
            radix_scatter_shader_->bind();
            if (indirect) glDispatchComputeIndirect(0);
            else glDispatchCompute(direct_wgs, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }

    void update_async_stats()
    {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        const uint32_t s = uint32_t(stats_frame_ % 3);
        glCopyNamedBufferSubData(gpu_state_buffer_, stats_buffer_, 0, s * 16, 16);
        if (stats_fence_[s]) glDeleteSync(stats_fence_[s]);
        stats_fence_[s] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        stats_frame_++;

        const uint32_t oldest = uint32_t(stats_frame_ % 3);
        if (stats_fence_[oldest])
        {
            const GLenum r = glClientWaitSync(stats_fence_[oldest], 0, 0);
            if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED)
            {
                const uint32_t * data = reinterpret_cast<const uint32_t *>(stats_ptr_) + size_t(oldest) * 4;
                stats_total_unclamped_ = data[0];
                stats_instances_ = data[1];
                if (stats_total_unclamped_ > instance_capacity_) allocate_instance_buffers(stats_total_unclamped_ + stats_total_unclamped_ / 2);
            }
        }
    }

    // Dimensions / state
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t tiles_x_ = 0;
    uint32_t tiles_y_ = 0;
    uint32_t tile_bits_ = 13;
    uint32_t num_gaussians_ = 0;
    uint32_t scene_sh_degree_ = 3;
    uint32_t instance_capacity_ = 0;
    bool cov3d_dirty_ = true;
    float last_scale_modifier_ = -1.0f;
    float last_frame_time_ms_ = 0.0f;

    // Async stats
    GLuint stats_buffer_ = 0;
    void * stats_ptr_ = nullptr;
    GLsync stats_fence_[3] {};
    uint64_t stats_frame_ = 0;
    uint32_t stats_instances_ = 0;
    uint32_t stats_total_unclamped_ = 0;

    // Static scene buffers
    gl_buffer pos_opacity_buffer_;
    gl_buffer rot_scale_buffer_;
    gl_buffer sh_buffer_;
    gl_buffer cov3d_buffer_;

    // Per-frame buffers
    gl_buffer tile_counts_buffer_;
    gl_buffer offsets_buffer_;
    gl_buffer attr_render_buffer_;
    gl_buffer attr_emit_buffer_;
    gl_buffer keys_buffer_[2];
    gl_buffer payloads_buffer_[2];
    gl_buffer hist_buffer_;
    gl_buffer scan_block_sums_buffer_;
    gl_buffer tile_boundary_buffer_;
    gl_buffer gpu_state_buffer_;
    gl_buffer indirect_buffer_;
    gl_buffer scan_len_counts_;
    gl_buffer scan_len_hist_;
    gl_buffer frame_ubo_;

    // Shaders
    std::unique_ptr<gl_shader_compute> precomp_cov3d_shader_;
    std::unique_ptr<gl_shader_compute> preprocess_shader_;
    std::unique_ptr<gl_shader_compute> scan_block_shader_;
    std::unique_ptr<gl_shader_compute> scan_sums_shader_;
    std::unique_ptr<gl_shader_compute> scan_apply_shader_;
    std::unique_ptr<gl_shader_compute> prepare_shader_;
    std::unique_ptr<gl_shader_compute> expand_shader_;
    std::unique_ptr<gl_shader_compute> radix_hist_shader_;
    std::unique_ptr<gl_shader_compute> radix_scatter_shader_;
    std::unique_ptr<gl_shader_compute> tile_boundary_shader_;
    std::unique_ptr<gl_shader_compute> render_shader_;
    GLint hist_shift_loc_ = -1;
    GLint scatter_shift_loc_ = -1;

    gpu_pass_timers timers_;
    gl_texture_2d output_texture_;
};

/////////////////////////////////
//   Viewer Harness Helpers    //
/////////////////////////////////

struct viewer_options
{
    std::string ply_path = "../assets/Placenta.ply";
    std::string screenshot_path;
    int benchmark_frames = 0;
    bool test_sort = false;
    bool test_scene = false;
    bool vsync = true;
};

inline viewer_options parse_viewer_options(int argc, char * argv[])
{
    viewer_options opts;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--benchmark" && i + 1 < argc) opts.benchmark_frames = std::stoi(argv[++i]);
        else if (arg == "--test-scene") opts.test_scene = true;
        else if (arg == "--test-sort") opts.test_sort = true;
        else if (arg == "--screenshot" && i + 1 < argc) opts.screenshot_path = argv[++i];
        else if (arg == "--vsync" && i + 1 < argc) opts.vsync = std::stoi(argv[++i]) != 0;
        else if (arg == "--ply" && i + 1 < argc) opts.ply_path = argv[++i];
    }
    return opts;
}

inline gaussian_vertex make_test_gaussian(const float3 & position, const float3 & color, const float sigma, const float opacity)
{
    constexpr float SH_C0 = 0.28209479177387814f;
    gaussian_vertex v {};
    v.position = float4(position, 1.0f);
    v.scale_opacity = float4(sigma, sigma, sigma, opacity);
    v.rotation = float4(1.0f, 0.0f, 0.0f, 0.0f); // identity, stored (w, x, y, z)
    std::memset(v.shs, 0, sizeof(v.shs));
    v.shs[0] = (color.x - 0.5f) / SH_C0;
    v.shs[1] = (color.y - 0.5f) / SH_C0;
    v.shs[2] = (color.z - 0.5f) / SH_C0;
    return v;
}

// Known-layout scene used by --test-scene to prove projection orientation and blend order.
// Camera is fixed at (0,0,4) looking at the origin. Expectations: red lands screen-right,
// green lands screen-top, and the near magenta splat wins the blend over the far yellow one
// (both lie on the same camera ray, so a backward sort shows yellow).
inline gaussian_splat_scene make_test_scene()
{
    gaussian_splat_scene scene;
    scene.sh_degree = 0;
    scene.vertices.push_back(make_test_gaussian({  0.0f,  0.0f, 0.0f }, { 0.1f, 0.1f, 1.0f }, 0.10f, 0.95f)); // blue center
    scene.vertices.push_back(make_test_gaussian({ +1.2f,  0.0f, 0.0f }, { 1.0f, 0.1f, 0.1f }, 0.10f, 0.95f)); // red +x
    scene.vertices.push_back(make_test_gaussian({  0.0f, +1.2f, 0.0f }, { 0.1f, 1.0f, 0.1f }, 0.10f, 0.95f)); // green +y
    scene.vertices.push_back(make_test_gaussian({ -1.5f,  0.0f, 0.0f }, { 1.0f, 1.0f, 0.1f }, 0.30f, 0.95f)); // yellow, far
    scene.vertices.push_back(make_test_gaussian({ -1.2f,  0.0f, 0.8f }, { 1.0f, 0.1f, 1.0f }, 0.10f, 0.95f)); // magenta, near, same ray
    return scene;
}

//////////////////////////////////////
//   Turntable Orbit Controller     //
//////////////////////////////////////

// Self-contained orbit: LMB drag rotates, RMB/MMB (or Shift+LMB) pans, wheel dollies.
class orbit_camera
{
    bool ml = false, mr = false, mm = false;
    float2 last_cursor {0, 0};
    bool has_cursor = false;

public:

    float3 target {0, 0, 0};
    float distance = 5.0f;
    float yaw = 0.0f;
    float pitch = 0.3f;

    void frame(const float3 & new_target, float new_distance)
    {
        target = new_target;
        distance = std::max(0.02f, new_distance);
    }

    void sync_from(const transform & pose, const float3 & pivot)
    {
        target = pivot;
        const float3 offset = pose.position - pivot;
        distance = std::max(0.02f, length(offset));
        pitch = std::asin(clamp(offset.y / distance, -1.0f, 1.0f));
        yaw = std::atan2(offset.x, offset.z);
    }

    transform pose() const
    {
        const float3 offset = {
            distance * std::cos(pitch) * std::sin(yaw),
            distance * std::sin(pitch),
            distance * std::cos(pitch) * std::cos(yaw) };
        return lookat_rh(target + offset, target);
    }

    void release_buttons() { ml = mr = mm = false; }

    void handle_input(const app_input_event & e)
    {
        if (e.type == app_input_event::MOUSE)
        {
            if (e.value[0] == GLFW_MOUSE_BUTTON_LEFT) ml = e.is_down();
            if (e.value[0] == GLFW_MOUSE_BUTTON_RIGHT) mr = e.is_down();
            if (e.value[0] == GLFW_MOUSE_BUTTON_MIDDLE) mm = e.is_down();
        }
        else if (e.type == app_input_event::SCROLL)
        {
            distance = std::max(0.02f, distance * std::pow(0.9f, float(e.value[1])));
        }
        else if (e.type == app_input_event::CURSOR)
        {
            if (!has_cursor) { last_cursor = e.cursor; has_cursor = true; return; }
            const float2 delta = e.cursor - last_cursor;
            last_cursor = e.cursor;

            const bool pan = mr || mm || (ml && (e.mods & GLFW_MOD_SHIFT));
            if (pan)
            {
                const transform p = pose();
                const float k = distance * 0.0015f;
                target += (p.xdir() * -delta.x + p.ydir() * delta.y) * k;
            }
            else if (ml)
            {
                yaw -= delta.x * 0.005f;
                pitch = clamp(pitch - delta.y * 0.005f, -1.55f, 1.55f);
            }
        }
    }
};

/////////////////////////////////
//   3DGS Viewer Application   //
/////////////////////////////////

struct gs_viewer_app final : public polymer_app
{
    perspective_camera cam;
    camera_controller_fps flycam;
    orbit_camera orbit;
    bool fly_mode = false;

    std::unique_ptr<gui::imgui_instance> igm;
    std::unique_ptr<simple_texture_view> fullscreen_surface;
    std::unique_ptr<gaussian_splat_renderer> renderer;

    gaussian_splat_scene scene; // fixme since this copies the gsplat
    std::string scene_filename;

    viewer_options opts;
    gs_render_settings settings;
    bool vsync_enabled = true;
    bool show_imgui = true;

    float3 scene_center {0, 0, 0};
    float scene_radius = 1.0f;

    // Harness state
    uint32_t harness_frame_index = 0;
    std::vector<float> bench_cpu_frame_ms;
    std::vector<float> bench_pass_ms[gs_pass_count];
    std::chrono::high_resolution_clock::time_point last_draw_timestamp;
    bool has_last_draw_timestamp = false;
    float cpu_frame_ema_ms = 0.0f;
    bool tests_failed = false;

    gs_viewer_app(const viewer_options & options) : polymer_app(1600, 1000, "Polymer 3DGS Viewer", 1), opts(options)
    {
        glfwMakeContextCurrent(window);
        vsync_enabled = opts.vsync && opts.benchmark_frames == 0 && !opts.test_scene && !opts.test_sort;
        glfwSwapInterval(vsync_enabled ? 1 : 0);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        // Load font
        auto droidSansTTFBytes = read_file_binary("../assets/fonts/droid_sans.ttf");
        igm.reset(new gui::imgui_instance(window));
        gui::make_light_theme();
        igm->add_font(droidSansTTFBytes);

        cam.pose = lookat_rh({ 0, 0, 4.f }, { 0, 0, 0 });
        cam.vfov = to_radians(45.0);
        cam.nearclip = 0.1f;
        cam.farclip = 1000.f;
        flycam.set_camera(&cam);

        renderer = std::make_unique<gaussian_splat_renderer>();
        renderer->initialize(width, height);

        fullscreen_surface = std::make_unique<simple_texture_view>();

        if (opts.test_sort)
        {
            run_sort_tests();
            exit();
        }
        else if (opts.test_scene)
        {
            scene = make_test_scene();
            scene_filename = "(test scene)";
            renderer->set_scene(scene);
            settings.sh_degree = int(scene.sh_degree);
            settings.flip_model = false; // test geometry is authored in GL space
            cam.pose = lookat_rh({ 0, 0, 4.f }, { 0, 0, 0 });
        }
        else
        {
            if (opts.benchmark_frames > 0) settings.flip_model = false; // match the baseline's identity model transform
            load_scene(opts.ply_path);
        }
    }

    void run_sort_tests()
    {
        struct { const char * name; uint32_t count; int pattern; } cases[] = {
            { "random-1M", 1000000, 0 },
            { "random-5M", 5000000, 0 },
            { "random-16M", 16000000, 0 },
            { "sorted-1M", 1000000, 1 },
            { "reverse-1M", 1000000, 2 },
            { "equal-1M", 1000000, 3 },
        };
        for (const auto & c : cases)
        {
            const bool ok = renderer->run_sort_test(c.count, c.pattern);
            std::cout << "TEST sort-" << c.name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
            if (!ok) tests_failed = true;
        }
    }

    void load_scene(const std::string & path)
    {
        if (!is_gaussian_splat_ply(path))
        {
            std::cerr << "Not a valid gaussian splat PLY file: " << path << std::endl;
            return;
        }

        scene = import_gaussian_splat_ply(path);
        scene_filename = get_filename_with_extension(path);

        if (scene.vertices.size() > 0)
        {
            renderer->set_scene(scene);
            settings.sh_degree = int(scene.sh_degree);
            compute_robust_bounds();
            reset_camera();
            glfwSetWindowTitle(window, ("Polymer 3DGS Viewer - " + scene_filename).c_str());
        }
    }

    // Median center + 95th-percentile radius: immune to the extreme outlier splats that
    // trained scenes accumulate (Placenta: r95 = 0.71 while max-distance radius is 240).
    void compute_robust_bounds()
    {
        std::vector<float> xs, ys, zs;
        const size_t stride = std::max<size_t>(1, scene.vertices.size() / 200000);
        for (size_t i = 0; i < scene.vertices.size(); i += stride)
        {
            xs.push_back(scene.vertices[i].position.x);
            ys.push_back(scene.vertices[i].position.y);
            zs.push_back(scene.vertices[i].position.z);
        }
        auto median = [](std::vector<float> & v) { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
        scene_center = float3(median(xs), median(ys), median(zs));

        std::vector<float> dist_sq;
        for (size_t i = 0; i < scene.vertices.size(); i += stride)
        {
            const float3 p(scene.vertices[i].position.x, scene.vertices[i].position.y, scene.vertices[i].position.z);
            dist_sq.push_back(length2(p - scene_center));
        }
        size_t k = std::min(size_t(dist_sq.size() * 0.95f), dist_sq.size() - 1);
        std::nth_element(dist_sq.begin(), dist_sq.begin() + k, dist_sq.end());
        scene_radius = std::max(0.01f, std::sqrt(dist_sq[k]));

        std::cout << "robust bounds: center=(" << scene_center.x << ", " << scene_center.y << ", " << scene_center.z
                  << ") radius=" << scene_radius << std::endl;
    }

    float framing_distance() const
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        const float aspect = height > 0 ? float(width) / float(height) : 1.6f;
        const float hfov = 2.0f * std::atan(std::tan(cam.vfov * 0.5f) * aspect);
        return 1.05f * scene_radius / std::sin(std::min(cam.vfov, hfov) * 0.5f);
    }

    // The model-flip rotates the scene 180 degrees about X, so the world-space pivot the
    // camera should orbit is the flipped center when the toggle is on.
    float3 display_center() const
    {
        return settings.flip_model ? float3(scene_center.x, -scene_center.y, -scene_center.z) : scene_center;
    }

    void reset_camera()
    {
        if (scene.vertices.empty()) return;
        orbit.yaw = 0.0f;
        orbit.pitch = 0.3f;
        orbit.frame(display_center(), framing_distance());
        cam.pose = orbit.pose();
        flycam.movementSpeed = std::max(1.0f, scene_radius * 2.5f);
        flycam.update_yaw_pitch();
    }

    void on_window_resize(int2 size) override
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width > 0 && height > 0) renderer->resize(width, height);
    }

    void on_input(const app_input_event & event) override
    {
        igm->update_input(event);

        if (ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard)
        {
            flycam.reset();
            orbit.release_buttons();
            return;
        }

        if (opts.benchmark_frames == 0 && !opts.test_scene)
        {
            if (fly_mode) flycam.handle_input(event);
            else orbit.handle_input(event);
        }

        if (event.type == app_input_event::KEY && event.action == GLFW_RELEASE)
        {
            if (event.value[0] == GLFW_KEY_R) reset_camera();
            if (event.value[0] == GLFW_KEY_TAB) show_imgui = !show_imgui;
            if (event.value[0] == GLFW_KEY_F12) take_screenshot();
            if (event.value[0] == GLFW_KEY_F)
            {
                fly_mode = !fly_mode;
                if (fly_mode) flycam.update_yaw_pitch();
                else orbit.sync_from(cam.pose, orbit.target);
            }
        }
    }

    void on_update(const app_update_event & e) override
    {
        if (opts.benchmark_frames > 0)
        {
            // Deterministic orbit around the robustly framed scene
            const float angle = (float(POLYMER_TAU) * harness_frame_index) / float(opts.benchmark_frames);
            const float3 offset = normalize(float3(std::sin(angle), 0.35f, std::cos(angle)));
            cam.pose = lookat_rh(display_center() + offset * framing_distance(), display_center());
        }
        else if (opts.test_scene)
        {
            // fixed camera
        }
        else if (fly_mode)
        {
            flycam.update(e.timestep_ms);
        }
        else
        {
            cam.pose = orbit.pose();
        }
    }

    void on_drop(std::vector<std::string> filepaths) override
    {
        for (const auto & path : filepaths)
        {
            if (get_extension(path) == "ply")
            {
                load_scene(path);
                break;
            }
        }
    }

    void on_draw() override
    {
        glfwMakeContextCurrent(window);

        const std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        if (has_last_draw_timestamp)
        {
            const float dt = std::chrono::duration<float, std::milli>(now - last_draw_timestamp).count();
            cpu_frame_ema_ms = cpu_frame_ema_ms <= 0.0f ? dt : (cpu_frame_ema_ms * 0.95f + dt * 0.05f);
            if (opts.benchmark_frames > 0 && harness_frame_index >= 10) bench_cpu_frame_ms.push_back(dt);
        }
        last_draw_timestamp = now;
        has_last_draw_timestamp = true;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width <= 0 || height <= 0) return;
        renderer->resize(width, height);

        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        renderer->render(cam, settings);
        fullscreen_surface->draw(renderer->get_output_texture());

        if (opts.benchmark_frames > 0 && harness_frame_index >= 10)
        {
            const float * pass_ms = renderer->get_pass_ms();
            for (uint32_t p = 0; p < gs_pass_count; ++p) bench_pass_ms[p].push_back(pass_ms[p]);
        }

        if (show_imgui && opts.benchmark_frames == 0 && !opts.test_scene)
        {
            igm->begin_frame();
            draw_ui();
            igm->end_frame();
        }

        glfwSwapBuffers(window);

        harness_frame_index++;

        if (opts.test_scene && harness_frame_index == 8)
        {
            run_test_probes(width, height);
            save_output_texture_png(opts.screenshot_path.empty() ? "3dgs-test-scene.png" : opts.screenshot_path, width, height);
            exit();
        }

        if (opts.benchmark_frames > 0 && harness_frame_index >= uint32_t(opts.benchmark_frames))
        {
            print_benchmark_stats();
            save_output_texture_png(opts.screenshot_path.empty() ? "3dgs-benchmark.png" : opts.screenshot_path, width, height);
            exit();
        }

        // Bare --screenshot: capture the default interactive view and exit
        if (opts.benchmark_frames == 0 && !opts.test_scene && !opts.screenshot_path.empty() && harness_frame_index == 8)
        {
            save_output_texture_png(opts.screenshot_path, width, height);
            exit();
        }
    }

    // Replicates the preprocess shader's world -> pixel mapping (including the +0.0001 w bias)
    // so probe locations always agree with where the renderer actually drew a splat center.
    int2 project_to_pixel(const float3 & world, int width, int height) const
    {
        const float aspect = float(width) / float(height);
        const float4 ph = (cam.get_projection_matrix(aspect) * cam.get_view_matrix()) * float4(world, 1.0f);
        const float pw = 1.0f / (ph.w + 0.0001f);
        const float2 ndc = { ph.x * pw, ph.y * pw };
        return { int(((ndc.x + 1.0f) * width - 1.0f) * 0.5f), int(((ndc.y + 1.0f) * height - 1.0f) * 0.5f) };
    }

    float3 sample_output_rgb(const std::vector<uint8_t> & rgba, int width, int height, int2 p) const
    {
        float3 sum {0, 0, 0};
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int x = p.x + dx, y = p.y + dy;
                if (x < 0 || y < 0 || x >= width || y >= height) continue;
                const size_t i = (size_t(y) * width + x) * 4;
                sum += float3(rgba[i] / 255.0f, rgba[i + 1] / 255.0f, rgba[i + 2] / 255.0f);
                count++;
            }
        }
        return count > 0 ? sum / float(count) : float3(0, 0, 0);
    }

    void check(const char * name, bool condition, const float3 & rgb)
    {
        std::cout << "TEST " << name << ": " << (condition ? "PASS" : "FAIL")
                  << " (r=" << rgb.x << " g=" << rgb.y << " b=" << rgb.z << ")" << std::endl;
        if (!condition) tests_failed = true;
    }

    void run_test_probes(int width, int height)
    {
        std::vector<uint8_t> rgba(size_t(width) * height * 4);
        glGetTextureImage(renderer->get_output_texture(), 0, GL_RGBA, GL_UNSIGNED_BYTE, GLsizei(rgba.size()), rgba.data());

        const int2 p_blue    = project_to_pixel({  0.0f,  0.0f, 0.0f }, width, height);
        const int2 p_red     = project_to_pixel({ +1.2f,  0.0f, 0.0f }, width, height);
        const int2 p_green   = project_to_pixel({  0.0f, +1.2f, 0.0f }, width, height);
        const int2 p_overlap = project_to_pixel({ -1.2f,  0.0f, 0.8f }, width, height);

        const float3 c_blue    = sample_output_rgb(rgba, width, height, p_blue);
        const float3 c_red     = sample_output_rgb(rgba, width, height, p_red);
        const float3 c_green   = sample_output_rgb(rgba, width, height, p_green);
        const float3 c_overlap = sample_output_rgb(rgba, width, height, p_overlap);

        std::cout << "probe pixels: blue=(" << p_blue.x << "," << p_blue.y << ") red=(" << p_red.x << "," << p_red.y
                  << ") green=(" << p_green.x << "," << p_green.y << ") overlap=(" << p_overlap.x << "," << p_overlap.y << ")" << std::endl;

        check("center-blue", c_blue.z > 0.4f && c_blue.z > c_blue.x + 0.1f && c_blue.z > c_blue.y + 0.1f, c_blue);
        check("red-color", c_red.x > 0.4f && c_red.x > c_red.y + 0.1f && c_red.x > c_red.z + 0.1f, c_red);
        check("green-color", c_green.y > 0.4f && c_green.y > c_green.x + 0.1f && c_green.y > c_green.z + 0.1f, c_green);

        // Orientation in texture space: +x world must land on the right half, +y world on the
        // top half (texture row index increases upward on screen with the non-flipping blit).
        check("red-screen-right", p_red.x > width / 2 && c_red.x > 0.4f, c_red);
        check("green-screen-top", p_green.y > height / 2 && c_green.y > 0.4f, c_green);

        // Blend order: near magenta must dominate far yellow along the shared camera ray. If the
        // sort order is backward (or broken), yellow dominates and green rises above the threshold.
        check("near-over-far", c_overlap.x > 0.35f && c_overlap.z > 0.25f && c_overlap.y < 0.3f, c_overlap);
    }

    void take_screenshot()
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        save_output_texture_png("3dgs-screenshot.png", width, height);
    }

    void save_output_texture_png(const std::string & path, int width, int height)
    {
        std::vector<uint8_t> rgba(size_t(width) * height * 4);
        glGetTextureImage(renderer->get_output_texture(), 0, GL_RGBA, GL_UNSIGNED_BYTE, GLsizei(rgba.size()), rgba.data());

        // Flip vertically: texture row 0 is the bottom of the screen, PNG row 0 is the top
        std::vector<uint8_t> flipped(rgba.size());
        for (int y = 0; y < height; ++y)
        {
            std::memcpy(flipped.data() + size_t(y) * width * 4, rgba.data() + size_t(height - 1 - y) * width * 4, size_t(width) * 4);
        }
        stbi_write_png(path.c_str(), width, height, 4, flipped.data(), width * 4);
        std::cout << "Saved screenshot: " << path << std::endl;
    }

    void print_benchmark_stats()
    {
        auto stats_line = [](const char * label, std::vector<float> values)
        {
            if (values.empty()) { printf("%-14s: no samples\n", label); return; }
            std::sort(values.begin(), values.end());
            float sum = 0.0f;
            for (float v : values) sum += v;
            const float avg = sum / values.size();
            const float p50 = values[values.size() / 2];
            const float p95 = values[size_t(values.size() * 0.95)];
            const float vmax = values.back();
            printf("%-14s: avg=%6.2f ms  p50=%6.2f ms  p95=%6.2f ms  max=%6.2f ms\n", label, avg, p50, p95, vmax);
        };

        std::cout << "=== 3DGS BENCHMARK (" << scene_filename << ", " << scene.vertices.size() << " gaussians, "
                  << bench_cpu_frame_ms.size() << " timed frames) ===" << std::endl;
        stats_line("cpu frame", bench_cpu_frame_ms);
        for (uint32_t p = 0; p < gs_pass_count; ++p) stats_line(gs_pass_name(p), bench_pass_ms[p]);
        std::cout << "instances (last frame): " << renderer->get_instance_count() << std::endl;
        if (!bench_cpu_frame_ms.empty())
        {
            float sum = 0.0f;
            for (float v : bench_cpu_frame_ms) sum += v;
            printf("avg fps: %.1f\n", 1000.0f * bench_cpu_frame_ms.size() / sum);
        }
    }

    void draw_ui()
    {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, 430), ImGuiCond_FirstUseEver);

        ImGui::Begin("Polymer 3DGS Viewer");

        ImGui::Text("Scene: %s", scene_filename.empty() ? "None (drop a .ply)" : scene_filename.c_str());
        ImGui::Text("Gaussians: %zu (SH degree %u)", scene.vertices.size(), scene.sh_degree);
        ImGui::Text("Tile instances: %u", renderer->get_instance_count());

        ImGui::Separator();

        const float * pass_ms = renderer->get_pass_ms();
        for (uint32_t p = gs_pass_preprocess; p < gs_pass_count; ++p)
        {
            ImGui::Text("%-12s %6.3f ms", gs_pass_name(p), pass_ms[p]);
        }
        ImGui::Text("%-12s %6.2f ms (%.0f fps)", "cpu frame", cpu_frame_ema_ms, cpu_frame_ema_ms > 0.0f ? 1000.0f / cpu_frame_ema_ms : 0.0f);

        ImGui::Separator();

        ImGui::SliderInt("SH Degree", &settings.sh_degree, 0, 3);
        ImGui::SliderFloat("Scale", &settings.scale_modifier, 0.1f, 3.0f);

        float fov_degrees = cam.vfov * 180.0f / float(POLYMER_PI);
        if (ImGui::SliderFloat("FOV", &fov_degrees, 30.0f, 90.0f)) cam.vfov = to_radians(fov_degrees);

        ImGui::ColorEdit3("Background", &settings.bg_color.x);

        if (ImGui::Checkbox("Flip Scene (COLMAP y-down)", &settings.flip_model))
        {
            reset_camera();
        }

        if (ImGui::Checkbox("VSync", &vsync_enabled))
        {
            glfwSwapInterval(vsync_enabled ? 1 : 0);
        }

        if (ImGui::Button("Reset View (R)")) reset_camera();
        ImGui::SameLine();
        if (ImGui::Button("Screenshot (F12)")) take_screenshot();

        ImGui::Separator();
        ImGui::TextWrapped(fly_mode
            ? "FLY mode (F to orbit): WASD moves, hold RMB + drag to look."
            : "ORBIT mode (F to fly): LMB drag rotates, RMB/MMB drag pans, wheel zooms.");

        ImGui::End();
    }
};

IMPLEMENT_MAIN(int argc, char * argv[])
{
    try
    {
        const viewer_options opts = parse_viewer_options(argc, argv);
        gs_viewer_app app(opts);
        app.main_loop();
        return app.tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    catch (const std::exception & e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
