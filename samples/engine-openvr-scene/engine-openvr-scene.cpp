#include "engine-openvr-scene.hpp"

renderable sample_vr_app::assemble_renderable(const entity e)
{
    renderable r;
    r.e = e;
    r.material = scene.render_system->get_material_component(e);
    r.mesh = scene.render_system->get_mesh_component(e);
    r.scale = scene.xform_system->get_local_transform(e)->local_scale;
    r.t = scene.xform_system->get_world_transform(e)->world_pose;
    return r;
}

sample_vr_app::sample_vr_app() : polymer_app(1280, 800, "sample-engine-openvr-scene")
{
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    desktop_imgui.reset(new gui::imgui_instance(window));
    gui::make_light_theme();

    try
    {
        hmd.reset(new openvr_hmd());
        glfwSwapInterval(0);

        orchestrator.reset(new entity_orchestrator());
        load_required_renderer_assets("../../assets/", shaderMonitor);

        shaderMonitor.watch("textured",
            "../../assets/shaders/renderer/forward_lighting_vert.glsl",
            "../../assets/shaders/renderer/textured_frag.glsl",
            "../../assets/shaders/renderer");

        scene.mat_library.reset(new polymer::material_library("../../assets/materials/"));

        // Setup for the recommended eye target size
        const uint2 eye_target_size = hmd->get_recommended_render_target_size();
        renderer_settings settings;
        settings.renderSize = int2(eye_target_size.x, eye_target_size.y);
        settings.cameraCount = 2;

        // Create required systems
        scene.collision_system = orchestrator->create_system<collision_system>(orchestrator.get());
        scene.xform_system = orchestrator->create_system<transform_system>(orchestrator.get());
        scene.identifier_system = orchestrator->create_system<identifier_system>(orchestrator.get());
        scene.render_system = orchestrator->create_system<render_system>(settings, orchestrator.get());

        teleporter.reset(new vr_teleport_system(orchestrator.get(), &scene, hmd.get()));

        vr_imgui.reset(new vr_imgui_surface(orchestrator.get(), &scene, { 256, 256 }, window));
        gui::make_light_theme();

        // Only need to set the skybox on the |render_payload| once (unless we clear the payload)
        payload.skybox = scene.render_system->get_skybox();
        payload.sunlight = scene.render_system->get_implicit_sunlight();

        {
            auto wf_mat = std::make_shared<polymer_wireframe_material>();
            scene.mat_library->create_material("renderer-wireframe", wf_mat);

            floor = scene.track_entity(orchestrator->create_entity());
            scene.identifier_system->create(floor, "floor-nav-mesh");
            scene.xform_system->create(floor, transform(make_rotation_quat_axis_angle({ 1, 0, 0 }, ((float) POLYMER_PI / 2.f)), { 0, -0.01f, 0 }), { 1.f, 1.f, 1.f });

            auto floor_geom = make_plane(48, 48, 24, 24);
            create_handle_for_asset("floor-mesh", make_mesh_from_geometry(floor_geom)); // gpu mesh

            polymer::material_component floor_mat(floor);
            floor_mat.material = material_handle("renderer-wireframe");
            scene.render_system->create(floor, std::move(floor_mat));

            polymer::mesh_component floor_mesh(floor);
            floor_mesh.mesh = gpu_mesh_handle("floor-mesh");
            scene.render_system->create(floor, std::move(floor_mesh));
        }

        // Setup left controller
        left_controller = scene.track_entity(orchestrator->create_entity());
        scene.identifier_system->create(left_controller, "openvr-left-controller");
        scene.xform_system->create(left_controller, transform(float3(0, 0, 0)), { 1.f, 1.f, 1.f });
        polymer::material_component left_material(left_controller);
        left_material.material = material_handle(material_library::kDefaultMaterialId);
        scene.render_system->create(left_controller, std::move(left_material));
        polymer::mesh_component left_mesh(left_controller);
        scene.render_system->create(left_controller, std::move(left_mesh));

        // Setup right controller
        right_controller = scene.track_entity(orchestrator->create_entity());
        scene.identifier_system->create(right_controller, "openvr-right-controller");
        scene.xform_system->create(right_controller, transform(float3(0, 0, 0)), { 1.f, 1.f, 1.f });
        polymer::material_component right_material(right_controller);
        right_material.material = material_handle(material_library::kDefaultMaterialId);
        scene.render_system->create(right_controller, std::move(right_material));
        polymer::mesh_component right_mesh(right_controller);
        scene.render_system->create(right_controller, std::move(right_mesh));

        // Setup render models for controllers when they are loaded
        hmd->controller_render_data_callback([this](cached_controller_render_data & data)
        {
            // We will get this callback for each controller, but we only need to handle it once for both.
            if (should_load == true)
            {
                should_load = false;

                // Create new gpu mesh from the openvr geometry
                auto mesh = make_mesh_from_geometry(data.mesh);
                create_handle_for_asset("openvr-controller-mesh", std::move(mesh));

                // Re-lookup components since they were std::move'd above
                auto lmc = scene.render_system->get_mesh_component(left_controller);
                assert(lmc != nullptr);

                auto rmc = scene.render_system->get_mesh_component(right_controller);
                assert(rmc != nullptr);

                // Set the handles
                lmc->mesh = gpu_mesh_handle("openvr-controller-mesh");
                rmc->mesh = gpu_mesh_handle("openvr-controller-mesh");
            }

        });
    }
    catch (const std::exception & e)
    {
        std::cout << "Application Init Exception: " << e.what() << std::endl;
    }

    // Setup left/right eye debug view we see on the desktop window
    eye_views.push_back(simple_texture_view()); // for the left view
    eye_views.push_back(simple_texture_view()); // for the right view
}

sample_vr_app::~sample_vr_app()
{
    hmd.reset();
}

void sample_vr_app::on_window_resize(int2 size)
{
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
}

void sample_vr_app::on_input(const app_input_event & event) 
{
    desktop_imgui->update_input(event);
}

void sample_vr_app::on_update(const app_update_event & e)
{
    frame_count++;

    shaderMonitor.handle_recompile();

    hmd->update();
    teleporter->update(frame_count);

    std::vector<openvr_controller::button_state> triggerStates = {
        hmd->get_controller(vr::TrackedControllerRole_LeftHand)->trigger,
        hmd->get_controller(vr::TrackedControllerRole_RightHand)->trigger
    };

    if (!scene.xform_system->set_local_transform(left_controller,
        hmd->get_controller(vr::TrackedControllerRole_LeftHand)->get_pose(hmd->get_world_pose()))) {
        std::cout << "Failed to set left controller transform..." << std::endl;
    }

    if (!scene.xform_system->set_local_transform(right_controller,
        hmd->get_controller(vr::TrackedControllerRole_RightHand)->get_pose(hmd->get_world_pose()))) {
        std::cout << "Failed to set right controller transform..." << std::endl;
    }

    // Billboard is on left hand
    auto lct = hmd->get_controller(vr::TrackedControllerRole_LeftHand)->get_pose(hmd->get_world_pose());
    lct = lct * transform(float4(0, 0, 0, 1), float3(0, 0, -.1f));
    lct = lct * transform(make_rotation_quat_axis_angle({ 1, 0, 0 }, (float) POLYMER_PI / 2.f), float3());
    lct = lct * transform(make_rotation_quat_axis_angle({ 0, 1, 0 }, (float) -POLYMER_PI), float3());

    auto rct = hmd->get_controller(vr::TrackedControllerRole_RightHand)->get_pose(hmd->get_world_pose());

    // Imgui needs the location of the pointer (controller), the billboard, and the click state.
    vr_imgui->update(&scene, rct, lct, triggerStates[1].pressed);
}

void sample_vr_app::on_draw()
{
    glfwMakeContextCurrent(window);

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    glViewport(0, 0, width, height);

    // Collect eye data for the render payload
    for (auto eye : { vr::Hmd_Eye::Eye_Left, vr::Hmd_Eye::Eye_Right })
    {
        const auto eye_pose = hmd->get_eye_pose(eye);
        const auto eye_projection = hmd->get_proj_matrix(eye, 0.075f, 64.f);
        payload.views.emplace_back(view_data(eye, eye_pose, eye_projection));
    }

    // Render scene using payload
    payload.render_set.clear();
    payload.render_set.push_back(assemble_renderable(left_controller));
    payload.render_set.push_back(assemble_renderable(right_controller));
    payload.render_set.push_back(assemble_renderable(floor));

    payload.render_set.push_back(assemble_renderable(vr_imgui->get_billboard()));

    if (vr_imgui->get_pointer() != kInvalidEntity)
    {
        payload.render_set.push_back(assemble_renderable(vr_imgui->get_pointer()));
    }

    if (teleporter->get_teleportation_arc() != kInvalidEntity)
    {
        payload.render_set.push_back(assemble_renderable(teleporter->get_teleportation_arc()));
    }

    glDisable(GL_CULL_FACE);
    scene.render_system->get_renderer()->render_frame(payload);

    const uint32_t left_eye_texture = scene.render_system->get_renderer()->get_color_texture(0);
    const uint32_t right_eye_texture = scene.render_system->get_renderer()->get_color_texture(1);

    // Render to the HMD
    hmd->submit(left_eye_texture, right_eye_texture);
    payload.views.clear();

    const aabb_2d rect{ { 0.f, 0.f },{ (float)width,(float)height } }; // Desktop window size
    const float mid = (rect.min().x + rect.max().x) / 2.f;
    const viewport_t leftViewport = { rect.min(),{ mid - 2.f, rect.max().y }, left_eye_texture };
    const viewport_t rightViewport = { { mid + 2.f, rect.min().y }, rect.max(), right_eye_texture };
    viewports.clear();
    viewports.push_back(leftViewport);
    viewports.push_back(rightViewport);

    if (viewports.size())
    {
        glUseProgram(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    // Draw to the desktop window
    for (int i = 0; i < viewports.size(); ++i)
    {
        const auto & v = viewports[i];
        glViewport(GLint(v.bmin.x), GLint(height - v.bmax.y), GLsizei(v.bmax.x - v.bmin.x), GLsizei(v.bmax.y - v.bmin.y));
        eye_views[i].draw(v.texture);
    }
    
    const auto headPose = hmd->get_hmd_pose();

    desktop_imgui->begin_frame();
    ImGui::Text("Head Pose: %f, %f, %f", headPose.position.x, headPose.position.y, headPose.position.z);
    desktop_imgui->end_frame();

    vr_imgui->begin_frame();
    gui::imgui_fixed_window_begin("controls", ui_rect{ {0, 0,}, {256, 256} });
    ImGui::Text("Head Pose: %f, %f, %f", headPose.position.x, headPose.position.y, headPose.position.z);
    ImGui::Text("Hit UV %f, %f", debug_pt.x, debug_pt.y);
    if (ImGui::Button("ImGui VR Button")) std::cout << "Click!" << std::endl;
    gui::imgui_fixed_window_end();
    vr_imgui->end_frame();

    // Update textures
    vr_imgui->update_renderloop();

    glfwSwapBuffers(window);

    gl_check_error(__FILE__, __LINE__);
}

int main(int argc, char * argv[])
{
    try
    {
        sample_vr_app app;
        app.main_loop();
    }
    catch (const std::exception & e)
    {
        POLYMER_ERROR("[Fatal] Caught exception: \n" << e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
