
# Import.
import sys 
import os 

# Blender imports.
import bpy
import bmesh
import mathutils
import math

# Numpy.
from numpy.random import default_rng

def create_instance(base_obj,
                    translate=mathutils.Vector((0,0,0)), 
                    scale=1.0,
                    rotate=("Z", 0.0),
                    basis=mathutils.Matrix.Identity(4),
                    tbn=mathutils.Matrix.Identity(4),
                    collection_name=None):
    # Create instance.
    inst_obj = bpy.data.objects.new(base_obj.name+"_inst", base_obj.data)
    # Perform translation, rotation, scaling and moving to target coord system for instance.
    mat_rot = mathutils.Matrix.Rotation(math.radians(rotate[1]), 4, rotate[0])
    mat_trans = mathutils.Matrix.Translation(translate)
    mat_sca = mathutils.Matrix.Scale(scale, 4) # TODO: scaling direction to input parameter! , (0,0,1)
    # TODO: If I am using `tbn` as basis then it sould go last, If I use `matrix_basis` as basis then it should go first.
    # `tbn` matrix is usually constructed for samples on base geometry using triangle normal. Therefore, it only contains
    # information about rotation.
    inst_obj.matrix_basis = basis @ mat_trans @ mat_rot @ mat_sca @ tbn  # TODO: is matrix_basis correct to be used for this?
    # Store to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(inst_obj)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(inst_obj)
    return inst_obj


def select_activate(obj):
    bpy.ops.object.select_all(action='DESELECT') # Deselect all objects
    bpy.context.view_layer.objects.active = obj   # Make given object the active object 
    obj.select_set(True)                          # Select given object

def unwrap_cube_project(object_to_unwrap, cube_size=0.3):
    select_activate(object_to_unwrap)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.cube_project(cube_size=cube_size, correct_aspect=True, clip_to_bounds=False, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')

def unwrap_smart_project(object_to_unwrap, angle_limit=66):
    select_activate(object_to_unwrap)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle_limit), island_margin=0.0, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')

# 
# Description:
# Create a cube using bmesh in center of world coordinate system.
# Apply translation, rotation and scaling using translate_vector, scale, rotation_axis_angle
# Apply rotation and translation to final coordinate system using basis
#
# Parameters:
# size - overall cube size
# translate - translation for given vector
# scale - scaling cube in specific vector direction
# rotate - rotate around given vector
# triangulate - triangulate quads into triangles
# basis - rotation and translation matrix defining target coordinate system
#
def create_cube(translate=mathutils.Vector((0,0,0)), 
                scale=1.0,
                rotate=("Z", 0.0),
                basis=mathutils.Matrix.Identity(4), # rotation and translation defining target (final) coordinate system
                tbn=mathutils.Matrix.Identity(4),
                collection_name=None): # collection destination
    # Create unit cube in center.
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=mathutils.Matrix.Identity(4), calc_uvs=False)
    object_mesh = bpy.data.meshes.new("cube_mesh")
    bm.to_mesh(object_mesh)
    bm.free()
    obj = bpy.data.objects.new("cube_obj", object_mesh)
    # Add transformation.
    mat_rot = mathutils.Matrix.Rotation(math.radians(rotate[1]), 4, rotate[0])
    mat_trans = mathutils.Matrix.Translation(translate)
    mat_sca = mathutils.Matrix.Scale(scale, 4)
    # TODO: If I am using tbn as basis then it sould go last, If I use matrix_basis as basis then it should go first?
    obj.matrix_basis = basis @ mat_trans @ mat_rot @ mat_sca @ tbn # TODO: is matrix_basis correct to be used? matrix_local vs matrix_parent_inverse vs matrix_world
    # Store to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(obj)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(obj)
    return obj


def create_icosphere(translate=mathutils.Vector((0,0,0)), 
                scale=1.0,
                rotate=("Z", 0.0),
                basis=mathutils.Matrix.Identity(4), # rotation and translation defining target (final) coordinate system
                tbn=mathutils.Matrix.Identity(4),
                collection_name=None):
    bm = bmesh.new()
    # Create icosphere.
    # https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_icosphere
    bmesh.ops.create_icosphere(bm, 
                subdivisions=1, 
                radius=1, 
                matrix=mathutils.Matrix.Identity(4), 
                calc_uvs=False)
    object_mesh = bpy.data.meshes.new("ico_sphere_mesh")
    bm.to_mesh(object_mesh)
    obj = bpy.data.objects.new("ico_sphere_obj", object_mesh)
    mat_rot = mathutils.Matrix.Rotation(math.radians(rotate[1]), 4, rotate[0])
    mat_trans = mathutils.Matrix.Translation(translate)
    mat_sca = mathutils.Matrix.Scale(scale, 4)
    # TODO: If I am using tbn as basis then it sould go last, If I use matrix_basis as basis then it should go first?
    obj.matrix_basis = basis @ mat_trans @ mat_rot @ mat_sca @ tbn # TODO: is matrix_basis correct to be used? matrix_local vs matrix_parent_inverse vs matrix_world
    # Store to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(obj)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(obj)
    return obj
    bm.free()
    return obj

# https://blender.stackexchange.com/questions/13986/how-to-join-objects-with-python
def fast_join(objects_to_join):
    context_tmp = bpy.context.copy()
    joined_object_name = objects_to_join[0].name # Joined object takes this name?
    context_tmp["active_object"] = objects_to_join[0] 
    context_tmp['selected_editable_objects'] = objects_to_join
    bpy.ops.object.join(context_tmp)
    return bpy.data.objects[joined_object_name]

# triangulate using bmesh.
def triangulate(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(obj.data)
    bm.free()

# https://stackoverflow.com/questions/19045971/random-rounding-to-integer-in-python
def probabilistic_round(x):
    return int(math.floor(x + mathutils.noise.random()))

# https://github.com/blender/blender/blob/master/source/blender/nodes/geometry/nodes/node_geo_distribute_points_on_faces.cc
# base_obj - MUST BE TRIANGULATED!
# returns: list of touples: (p, N, w, tbn)
def mesh_uniform_weighted_sampling(base_obj, n_samples, base_density=1.0, use_weight_paint=False, min_distance=0.0):
    triangulate(base_obj)
    rng = default_rng()
    samples = [] # (p, N, w, tbn)
    samples_all = []
    for polygon in base_obj.data.polygons: # must be triangulated mesh!
        # Extract triangle vertices and their weights.
        triangle_vertices = []
        triangle_vertex_weights = []
        for v_idx in polygon.vertices:
            v = base_obj.data.vertices[v_idx]
            triangle_vertices.append(v.co)
            if len(v.groups) < 1:
                triangle_vertex_weights.append(0.0)
            else:
                triangle_vertex_weights.append(v.groups[0].weight) # TODO: only one group? Investigate! float in [0, 1], default 0.0
        # Create samples.
        polygon_density = 1
        if use_weight_paint:
            polygon_density = (triangle_vertex_weights[0] + triangle_vertex_weights[1] + triangle_vertex_weights[2]) / 3.0
            if polygon_density < 1e-4:
                polygon_density = 0.0
        point_amount = probabilistic_round(polygon.area * polygon_density * base_density)
        for i in range(point_amount):
            a = mathutils.noise.random()
            b = mathutils.noise.random()
            c = mathutils.noise.random()
            s = a + b + c
            un = (a / s)
            vn = (b / s)
            wn = (c / s)
            p = un * triangle_vertices[0] + vn * triangle_vertices[1] + wn * triangle_vertices[2]
            w = un * triangle_vertex_weights[0] + vn * triangle_vertex_weights[1] + wn * triangle_vertex_weights[2] # interpolate weight
            n = polygon.normal # TODO: vertex normals?
            # Calc BTN. 
            t = mathutils.Vector(triangle_vertices[0] - p) # NOTE: use most distant point from barycentric coord to evade problems with 0
            t = t.normalized()
            bt = n.cross(t)
            bt = bt.normalized()
            tbn = mathutils.Matrix((t, bt, n)) # NOTE: using pixar_onb()?
            tbn = tbn.transposed() # TODO: why transposing?
            tbn.resize_4x4()
            samples_all.append([p,n,w,tbn])
    random_sample_indices = rng.integers(len(samples_all), size=n_samples)
    for i in random_sample_indices:
        samples.append(samples_all[i])
    return samples
    

# Create points in cube volume.
def sample_points_in_cube(n_points, cube_x, cube_y, cube_z, origin=mathutils.Vector((0,0,0))):
    samples = []
    for i in range(n_points):
        x = mathutils.noise.random() * cube_x
        y = mathutils.noise.random() * cube_y
        z = mathutils.noise.random() * cube_z
        samples.append(mathutils.Vector((x, y, z))+origin)
    return samples

def line_guided_sampling(line, volume_radius, n_volume_points):
    samples = []
    for edge in line.data.edges:
        start_vert = line.data.vertices[edge.vertices[0]]
        end_vert = line.data.vertices[edge.vertices[1]]
        edge_vec = mathutils.Vector(end_vert.co - start_vert.co)
        edge_dir = edge_vec.normalized()
        edge_len = edge_vec.length
        n_samples = math.ceil(edge_len / (volume_radius/2.0))
        curr_sample_pos = mathutils.Vector(start_vert.co) # NOTE: create new object, do not work with reference!
        for sample in range(n_samples):
            curr_sample_pos += volume_radius * (sample/n_samples) * edge_dir
            #bpy.ops.mesh.primitive_uv_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=curr_sample_pos, scale=(1, 1, 1))
            local_samples = sample_points_in_cube(n_volume_points, volume_radius, volume_radius, volume_radius, curr_sample_pos)
            samples.extend(local_samples)
    return samples

def create_on_points(samples, rsx=[0.5,1.0], rsy=[0.5,1.0], rsz=[0.5,1.0]):
    created_elements = []
    for sample in samples:
        if mathutils.noise.random() > 0.5:
            base_elem = create_icosphere(translate=sample)
        else:
            base_elem = create_cube(translate=sample)
        sx = (rsx[1] - rsx[0]) *  mathutils.noise.random() + rsx[0]
        sy = (rsy[1] - rsy[0]) *  mathutils.noise.random() + rsy[0]
        sz = (rsz[1] - rsz[0]) *  mathutils.noise.random() + rsz[0]
        base_elem.scale = mathutils.Vector((sx, sy, sz))
        created_elements.append(base_elem)
    return created_elements


def apply_location_rotation_scale(obj, location=True, rotation=True, scale=True):
    select_activate(obj)
    bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)

def bevel(object_to_bevel, amount=0.2, segments=3):
    bevel_mod = object_to_bevel.modifiers.new("Bevel", 'BEVEL')
    bevel_mod.width = amount
    bevel_mod.segments = segments
    select_activate(object_to_bevel)
    bpy.ops.object.modifier_apply(modifier="Bevel")

def decimate(object_to_decimate, ratio=0.8):
    decimate_mod = object_to_decimate.modifiers.new("Decimate", 'DECIMATE')
    decimate_mod.ratio = ratio
    select_activate(object_to_decimate)
    bpy.ops.object.modifier_apply(modifier="Decimate")

def remesh_blocks(object_to_remesh, octree_depth):
    remesh_mod = object_to_remesh.modifiers.new("Remesh", 'REMESH')
    remesh_mod.mode = "BLOCKS"
    remesh_mod.octree_depth = octree_depth
    select_activate(object_to_remesh)
    bpy.ops.object.modifier_apply(modifier="Remesh")

def remesh_smooth(object_to_remesh, octree_depth):
    remesh_mod = object_to_remesh.modifiers.new("Remesh", 'REMESH')
    remesh_mod.mode = "SMOOTH"
    remesh_mod.octree_depth = octree_depth
    select_activate(object_to_remesh)
    bpy.ops.object.modifier_apply(modifier="Remesh")

def remesh_voxel(object_to_remesh, voxel_size=0.25, adaptivity=0.1):
    remesh_mod = object_to_remesh.modifiers.new("Remesh", 'REMESH')
    remesh_mod.mode = "VOXEL"
    remesh_mod.voxel_size = voxel_size
    remesh_mod.adaptivity = adaptivity
    select_activate(object_to_remesh)
    bpy.ops.object.modifier_apply(modifier="Remesh")

def create_clouds_tex(noise_scale=1.8, noise_depth=3):
    tex = bpy.data.textures.new("clouds_tex", "CLOUDS")
    tex.noise_scale = noise_scale
    tex.noise_depth = noise_depth
    return tex

def displace(object_to_displace, displace_strength=0.5, clouds_noise_scale=1.8, clouds_noise_depth=3):
    displace_mod = object_to_displace.modifiers.new("Displace", "DISPLACE")
    clouds_tex = create_clouds_tex(noise_scale=clouds_noise_scale, noise_depth=clouds_noise_depth)
    displace_mod.texture = clouds_tex
    displace_mod.strength = displace_strength
    select_activate(object_to_displace)
    bpy.ops.object.modifier_apply(modifier="Displace")

def create_rock(rock_elements, rock_name="rock"):
    # join base elements. NOTE: here we need actual geometry not only instances!
    rock_elements_joined = fast_join(rock_elements)
    # TODO: CSG: generate elongated cubes on vertices in normal direction, use difference to make holes
    # Bevel
    #bevel(object_to_bevel=rock_elements_joined, amount=0.2, segments=3)
    # remesh
    #remesh_blocks(object_to_remesh=rock_elements_joined, octree_depth=6)
    # remesh
    remesh_smooth(object_to_remesh=rock_elements_joined, octree_depth=7)
    # remesh
    remesh_voxel(object_to_remesh=rock_elements_joined, voxel_size=0.2, adaptivity=0.3)
    # remesh
    remesh_voxel(object_to_remesh=rock_elements_joined, voxel_size=0.05, adaptivity=0.05)
    # apply scale.
    apply_location_rotation_scale(rock_elements_joined)
    # TODO: subdivision surface?
    # displace
    displace(object_to_displace=rock_elements_joined, displace_strength=0.5, clouds_noise_scale=2, clouds_noise_depth=3)
    # decimate
    decimate(object_to_decimate=rock_elements_joined, ratio=0.5)
    # unwrap
    unwrap_smart_project(object_to_unwrap=rock_elements_joined, angle_limit=66)
    #unwrap_cube_project(object_to_unwrap=rock_elements_joined, cube_size=0.2)
    # TODO: texture and material: diff, rough, normal, bump
    rock_elements_joined.name = rock_name
    triangulate(rock_elements_joined)
    return rock_elements_joined

#
# Specific use cases.
#

def line_guided_elements(cluster_radius=5, cluster_samples=10, name="line_guided_rock"):
    # Select line (edges and vertices of selected object).
    base_obj = bpy.context.selected_objects[0]
    # Sample points on lines and for each point sample points in cube positioned on that sample with radius local_cluster_radius.
    local_cluster_radius = cluster_radius
    samples = line_guided_sampling(line=base_obj, volume_radius=local_cluster_radius, n_volume_points=cluster_samples)
    rsx = [local_cluster_radius - local_cluster_radius / 2.0, local_cluster_radius + local_cluster_radius / 2.0]
    rsy = [local_cluster_radius - local_cluster_radius / 2.0, local_cluster_radius + local_cluster_radius / 2.0]
    rsz = [local_cluster_radius - local_cluster_radius / 2.0, local_cluster_radius + local_cluster_radius / 2.0]
    # Create base elements on samples.
    created_elements = create_on_points(samples, rsx=rsx, rsy=rsy, rsz=rsz)
    joined_created_elements = fast_join(created_elements)
    return joined_created_elements

# Create rock using line guiding.
def line_guided_rock(cluster_radius=5, cluster_samples=10, name="line_guided_rock"):
    # Select line (edges and vertices of selected object).
    base_obj = bpy.context.selected_objects[0]
    # Sample points on lines and for each point sample points in cube positioned on that sample with radius local_cluster_radius.
    local_cluster_radius = cluster_radius
    samples = line_guided_sampling(line=base_obj, volume_radius=local_cluster_radius, n_volume_points=cluster_samples)
    rsx = [local_cluster_radius - local_cluster_radius / 2.0, local_cluster_radius + local_cluster_radius / 2.0]
    rsy = [local_cluster_radius - local_cluster_radius / 2.0, local_cluster_radius + local_cluster_radius / 2.0]
    rsz = [local_cluster_radius - local_cluster_radius / 2.0, local_cluster_radius + local_cluster_radius / 2.0]
    # Create base elements on samples.
    created_elements = create_on_points(samples, rsx=rsx, rsy=rsy, rsz=rsz)
    # Create rock using base samples.
    line_guided_rock = create_rock(created_elements, name)
    return line_guided_rock

def mesh_sampling_rock(n_samples=30, base_density=15.0, use_weight_paint=False, name="mesh_sampling_rock"):
    # Get selected object.
    base_obj = bpy.context.selected_objects[0]
    # Sample selected object.
    samples = mesh_uniform_weighted_sampling(base_obj, n_samples=n_samples, base_density=base_density, use_weight_paint=use_weight_paint, min_distance=0.0)
    # Create elements on samples.
    rsx = [base_obj.dimensions[0]/4.0, base_obj.dimensions[0]/2.0]
    rsy = [base_obj.dimensions[1]/4.0, base_obj.dimensions[1]/2.0]
    rsz = [base_obj.dimensions[2]/4.0, base_obj.dimensions[2]/2.0]
    created_elements = create_on_points(samples, rsx=rsx, rsy=rsy, rsz=rsz)
    # Create rock from elements.
    mesh_sampling_rock = create_rock(created_elements, "mesh_sampling_rock")
    return mesh_sampling_rock

def cube_sampling_rock(cube_x=2, cube_y=3, cube_z=10, origin=mathutils.Vector((0,0,0)), name="cube_volume_rock"):
    n_samples = 20
    samples = sample_points_in_cube(n_points=n_samples, cube_x=cube_x, cube_y=cube_y, cube_z=cube_z, origin=origin)
    rsx = [cube_x/4.0, cube_x/2.0]
    rsy = [cube_y/4.0, cube_y/2.0]
    rsz = [cube_z/4.0, cube_z/2.0]
    created_elements = create_on_points(samples, rsx=rsx, rsy=rsy, rsz=rsz)
    cube_volume_rock = create_rock(created_elements, name)
    return cube_volume_rock


def main():
    #
    # Line guided rock creation.
    #
    line_guided_rock(cluster_radius=5, cluster_samples=5)
    #line_guided_elements(cluster_radius=5, cluster_samples=5)

    #
    # Cube sampling rock creation.
    #
    #cube_sampling_rock(cube_x=2, cube_y=3, cube_z=10, origin=mathutils.Vector((0,0,0)), name="cube_volume_rock")
    

    #
    # Mesh sampling rock creation.
    #
    #mesh_sampling_rock(n_samples=30, base_density=15.0, use_weight_paint=False, name="mesh_sampling_rock")
    

    #
    # Instance rocks on mesh.
    #
    # First, create rock base elements.
    """
    base_elems = []
    n_samples = 20
    horizontal_sampling = [10, 5, 1]
    horizontal_scale = [[2.0, 3.0], [1.3, 1.5], [0.6, 0.7]]
    samples = sample_points_in_cube(n_points=n_samples, cube_x=horizontal_sampling[0], cube_y=horizontal_sampling[1], cube_z=horizontal_sampling[2], origin=mathutils.Vector((0,0,0)))
    created_elements = create_on_points(samples, rsx=horizontal_scale[0], rsy=horizontal_scale[1], rsz=horizontal_scale[2])
    rock1 = create_rock(created_elements, "rock1")
    base_elems.append(rock1)
    vertical_sampling = [3, 3, 10]
    vertical_scale = [[1.3, 1.5], [1.3, 1.5], [2.0, 3.0]]
    samples = sample_points_in_cube(n_points=n_samples, cube_x=vertical_sampling[0], cube_y=vertical_sampling[1], cube_z=vertical_sampling[2], origin=mathutils.Vector((0,0,0)))
    created_elements = create_on_points(samples, rsx=vertical_scale[0], rsy=vertical_scale[1], rsz=vertical_scale[2])
    rock2 = create_rock(created_elements, "rock2")
    base_elems.append(rock2)
    # Sample selected object.
    base_obj = bpy.context.selected_objects[0]
    base_obj_samples = mesh_uniform_weighted_sampling(base_obj=base_obj, n_samples=100, base_density=5.0, use_weight_paint=True)
    # Instance.
    for sample in base_obj_samples:
        create_instance(rock1,
                        translate=sample[0], 
                        scale=2,
                        rotate=("Z", 0.0),
                        basis=mathutils.Matrix.Identity(4),
                        tbn=sample[3],
                        collection_name=None)
    """


if __name__ == "__main__":
    main()
