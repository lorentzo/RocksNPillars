
# Blender 3.1.

# Add details to pillars: displacement, bevel, textures
# Create multiple base pillars with different details
# Instance pillars using different base shapes

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
    mat_sca = mathutils.Matrix.Scale(scale, 4, (0,0,1)) # TODO: scaling direction to input parameter! , (0,0,1)
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

def unwrap_smart_project(object_to_unwrap, angle_limit=66):
    select_activate(object_to_unwrap)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle_limit), island_margin=0.0, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')

def unwrap_cube_project(object_to_unwrap, cube_size=0.3):
    select_activate(object_to_unwrap)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.cube_project(cube_size=cube_size, correct_aspect=True, clip_to_bounds=False, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')

def create_pillar(segments=6, size=1.0, obj_name="pillar", collection_name=None):
    # Create bmesh pillar.
    bm = bmesh.new()
    #bmesh.ops.create_circle(bm, cap_ends=True, cap_tris=False, segments=segments, radius=size, matrix=mathutils.Matrix.Identity(4), calc_uvs=False) # TODO: manual: https://www.redblobgames.com/grids/hexagons/#angles
    # Extrude faces in normal direction.
    #efaces = bmesh.ops.extrude_discrete_faces(bm, faces=bm.faces)
    #for eface in efaces["faces"]:
    #    bmesh.ops.translate(bm,verts=eface.verts,vec=eface.normal*1.0) # TODO: bottom doesn't have a face now!
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=size, radius2=size, depth=1, matrix=mathutils.Matrix.Identity(4), calc_uvs=True)
    # Subdivide.
    # https://blender.stackexchange.com/questions/120242/when-i-subdivide-a-face-how-do-i-get-each-face-from-the-result-in-python
    # use_grid_fill subdivides the faces!
    #bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=5, use_grid_fill=True)
    # Bevel.
    bmesh.ops.bevel(bm, geom=(bm.edges), offset=0.3, affect="VERTICES", segments=3, profile=0.5)
    
    for v in bm.verts:
        #offset_amount = mathutils.noise.fractal(v.co * mathutils.noise.random()*20.0, 15, 13, 14, noise_basis='PERLIN_ORIGINAL') * 0.05
        #offset_amount = mathutils.noise.hetero_terrain(v.co * mathutils.noise.random()*20.0, 15, 13, 14, 0, noise_basis='PERLIN_ORIGINAL') * 0.05
        #offset_amount = mathutils.noise.random() * 0.05
        offset_dir = v.normal# * offset_amount
        #bmesh.ops.translate(bm, vec=offset_dir, space=mathutils.Matrix.Identity(4), verts=[v], use_shapekey=False)
    
    # Create mesh and object from bmesh.
    object_mesh = bpy.data.meshes.new(obj_name+"_mesh")
    bm.to_mesh(object_mesh)
    obj = bpy.data.objects.new(obj_name, object_mesh)
    #triangulate(obj)
    # Store object to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(obj)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(obj)
    # Unwrap.
    unwrap_smart_project(obj, angle_limit=66)
    return obj

def create_pillar2(segments=6, size=1.0, obj_name="pillar", collection_name=None):
    # Create bmesh pillar.
    bm = bmesh.new()
    #bmesh.ops.create_circle(bm, cap_ends=True, cap_tris=False, segments=segments, radius=size, matrix=mathutils.Matrix.Identity(4), calc_uvs=False) # TODO: manual: https://www.redblobgames.com/grids/hexagons/#angles
    # Extrude faces in normal direction.
    #efaces = bmesh.ops.extrude_discrete_faces(bm, faces=bm.faces)
    #for eface in efaces["faces"]:
    #    bmesh.ops.translate(bm,verts=eface.verts,vec=eface.normal*1.0) # TODO: bottom doesn't have a face now!
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=True, segments=segments, radius1=size, radius2=size, depth=1, matrix=mathutils.Matrix.Identity(4), calc_uvs=True)
    # Subdivide.
    # https://blender.stackexchange.com/questions/120242/when-i-subdivide-a-face-how-do-i-get-each-face-from-the-result-in-python
    # use_grid_fill subdivides the faces!
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=5, use_grid_fill=True)    
    for v in bm.verts:
        #offset_amount = mathutils.noise.fractal(v.co * mathutils.noise.random()*20.0, 15, 13, 14, noise_basis='PERLIN_ORIGINAL') * 0.05
        #offset_amount = mathutils.noise.hetero_terrain(v.co * mathutils.noise.random()*20.0, 15, 13, 14, 0, noise_basis='PERLIN_ORIGINAL') * 0.05
        offset_amount = mathutils.noise.random() * 0.01
        offset_dir = v.normal * offset_amount
        bmesh.ops.translate(bm, vec=offset_dir, space=mathutils.Matrix.Identity(4), verts=[v], use_shapekey=False)
    
    # Create mesh and object from bmesh.
    object_mesh = bpy.data.meshes.new(obj_name+"_mesh")
    bm.to_mesh(object_mesh)
    obj = bpy.data.objects.new(obj_name, object_mesh)
    #triangulate(obj)
    # Store object to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(obj)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(obj)
    # Unwrap.
    unwrap_cube_project(obj)
    return obj

# https://www.redblobgames.com/grids/hexagons/#coordinates-offset
def create_hex_grid_centers(hex_size, grid_size_x, grid_size_y):
    centers = []
    center = mathutils.Vector((0.0, 0.0, 0.0))
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            center_curr_x = center[0] + i * math.sqrt(3.0) * hex_size
            if j % 2 != 0:
                center_curr_x = center_curr_x + (math.sqrt(3.0) * hex_size) / 2.0
            center_curr_y = center[1] + j * (3.0 / 4.0) * 2.0 * hex_size
            centers.append(mathutils.Vector((center_curr_x, center_curr_y, 0.0)))
    return centers

# Imagine that infinite hex grid exists, for given point check in which hex grid coordinate it falls into.
# https://stackoverflow.com/questions/7705228/hexagonal-grids-how-do-you-find-which-hexagon-a-point-is-in
def hex_grid_coordinate(sample, grid_size_x, grid_size_y):
    # Row and column of box that sample falls in.
    row = int(sample[1]) / grid_size_y
    row_is_odd = row % 2 == 1
    if row_is_odd:
        column = int( (x - (grid_size_x/2) / grid_size_x) )
    else:
        column = int(x / grid_size_x)
    # Find out the position of the point relative to the box it is in.
    rel_y = point[1] - (row * grid_size_y)
    if (row_is_odd):
        rel_x = (x - (column * grid_size_x) - (grid_size_x/2))
    else:
        rel_x = x - (column * grid_size_x)
    # TODO: finish....

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

# https://blender.stackexchange.com/questions/220072/check-using-name-if-a-collection-exists-in-blend-is-linked-to-scene
def create_collection_if_not_exists(collection_name):
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection) #Creates a new collection


def main():
    base_obj = bpy.context.selected_objects[0]
    pillar_centers = mesh_uniform_weighted_sampling(base_obj=base_obj, n_samples=500, base_density=50.0, use_weight_paint=True, min_distance=0.0)
    hex_size = 1.0
    base_pillars = []
    hex_pillar_bevel = create_pillar(segments=6, size=hex_size, obj_name="pillar", collection_name=None)
    base_pillars.append(hex_pillar_bevel)
    hex_pillar_rand1 = create_pillar2(segments=6, size=hex_size, obj_name="pillar", collection_name=None)
    base_pillars.append(hex_pillar_rand1)
    hex_pillar_rand2 = create_pillar2(segments=6, size=hex_size, obj_name="pillar", collection_name=None)
    base_pillars.append(hex_pillar_rand2)
    
    #hex_grid_size_x = math.ceil(base_obj.dimensions[0] / hex_pillar.dimensions[0])
    #hex_grid_size_y = math.ceil(base_obj.dimensions[1] / hex_pillar.dimensions[1])
    #print(hex_grid_size_x, hex_grid_size_y)
    #pillar_centers = create_hex_grid_centers(hex_size=hex_size, grid_size_x=hex_grid_size_x, grid_size_y=hex_grid_size_y)
    for pillar_center in pillar_centers:
        scale = (10 - 5) * mathutils.noise.random() + 5
        rng = default_rng()
        random_sample_indices = rng.integers(len(base_pillars), size=1)[0]
        create_instance(base_pillars[random_sample_indices],
                        translate=pillar_center[0], 
                        scale=scale,
                        rotate=("Z", 0.0),
                        basis=mathutils.Matrix.Identity(4),
                        tbn=mathutils.Matrix.Identity(4),
                        collection_name=None)

if __name__ == "__main__":
    main()
