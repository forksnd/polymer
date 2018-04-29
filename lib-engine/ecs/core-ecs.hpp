#pragma once

#include "typeid.hpp"

#ifndef polymer_entity_component_system_hpp
#define polymer_entity_component_system_hpp

#include <unordered_map>

using namespace polymer;

////////////////
//   Entity   //
////////////////

// An entity is an uniquely identifiable object in the Polymer runtime.
using entity = uint64_t;
constexpr entity kInvalidEntity = 0;

////////////////////////
//   Base Component   //
////////////////////////

// Provide a consistent way to retrieve an entity to which a component belongs. 
class base_component
{
    entity e;
    friend struct component_hash;
public:
    explicit base_component(entity e = kInvalidEntity) : e(e) {}
    entity get_entity() const { return e; }
};

// Hash functor for components so they can be used in unordered containers. 
struct component_hash { entity operator()(const base_component & c) const { return c.e; } };

/////////////////////
//   Base System   //
/////////////////////

// Systems are responsible for storing the component data instances associated with Entities.
// They also perform all the logic for manipulating and processing their Components.
// This base class provides an API for an entity_manager to associate Components with Entities in a data-driven manner.

class entity_orchestrator;
struct base_system : public non_copyable
{
    entity_orchestrator * orchestrator;

    explicit base_system(entity_orchestrator * o) : orchestrator(o) {}
    virtual ~base_system() {}

    // Associates component with the Entity using serialized data. The void pointer 
    // and hash type is to subvert the need for a heavily templated component system. 
    virtual bool create(entity e, poly_typeid hash, void * data) = 0;

    // Destroys all of an entity's associated components
    virtual void destroy(entity e) = 0;

    // Helper function to signal to the entity manager that this system operates on these types of components
    template <typename S>
    void register_system_for_type(S * system, poly_typeid type) { register_system_for_type(get_typeid<S>(), type); }
    void register_system_for_type(poly_typeid system_type, poly_typeid type);
};

/////////////////////////////
//   Entity Orchestrator   //
/////////////////////////////

class entity_orchestrator
{
    std::mutex createMutex;
    std::unordered_map<poly_typeid, poly_typeid> system_type_map;
    std::unordered_map<poly_typeid, base_system *> systems;
    entity entity_counter{ 0 }; // Autoincrementing value to generate unique ids.

public:

    template <typename T, typename... Args>
    T * create_system(Args &&... args)
    {
        T * ptr = new T(std::forward<Args>(args)...);
        add_system(get_typeid<T>(), ptr);
        return ptr;
    }

    void register_system_for_type(poly_typeid system_type, poly_hash_value def_type)
    {
        system_type_map[def_type] = system_type;
    }

    entity create_entity()
    {
        std::lock_guard<std::mutex> guard(createMutex);
        const entity e = ++entity_counter;
        return e;
    }

    void add_system(poly_typeid system_type, base_system * system)
    {
        if (!system) return;
        auto itr = systems.find(system_type);
        if (itr == systems.end()) systems.emplace(system_type, system);
    }
};

void base_system::register_system_for_type(poly_typeid system_type, poly_typeid entity_type) { orchestrator->register_system_for_type(system_type, entity_type); }

#endif // polymer_entity_component_system_hpp