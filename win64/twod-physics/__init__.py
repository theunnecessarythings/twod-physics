import sys
import os
path = os.path.dirname(os.path.realpath(__file__))
print(path)
sys.path.append(os.path.join(path, 'modules'))
print(sys.path[-1])
import bpy
import bgl
import blf
import gpu

from gpu_extras.batch import batch_for_shader


from bpy.types import (
    Operator,
    Panel,
    PropertyGroup,
)
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    CollectionProperty
)

import math
import random
import uuid
''' All physics stuff goes here '''

import pymunk
from pymunk import Vec2d
from pyhull.convex_hull import ConvexHull
bl_info = {
   'name': 'TwoD Physics',
   'author': 'Sreeraj R',
   'version': (1, 0, 0),
   'blender': (2, 80, 0),
   "description": "2D Physics Engine for Blender",
   'location': 'In Physics tab and Scene Tab',
   "warning": "",
   "wiki_url": "",
   "tracker_url": "",
   'category': 'Physics',
}

COLOR_WORLD = (1, 1, 1, 1)
world = None
class World:
    def __init__(self, context):
        self.space = None
        self.props = context.scene.twod_props
        self.bodies = set()
        self.constraints = set()
        self.collision_shapes = {}
        self.rigid_bodies = {}
        self.constraints_dict = {}
        self.create_world(context)
        self.create_constraints(context)
        global _handle_3d
        _handle_3d = bpy.types.SpaceView3D.draw_handler_add(draw_callback_3d, (self, context), 'WINDOW', 'POST_VIEW')
    
        
    def create_world(self, context):
        self.clean_world()
        self.space = pymunk.Space()
        self.space.gravity = self.props.world_gravity[:2]   
        self.space.iterations = self.props.iterations
        self.space.collision_bias = self.props.collision_bias
        self.space.collision_persistence = self.props.collision_persistence
        self.space.collision_slop = self.props.collision_slop
        self.space.damping = self.props.world_damping

        
        self.update_bodies(context)
        for obj in self.bodies:
            obj_props = obj.twod_object_props
            mass = obj_props.object_mass
            moment_type = obj_props.object_moi_type
            radius = obj_props.object_radius
            inner_radius = obj_props.object_inner_radius
            outer_radius = obj_props.object_outer_radius
            offset = obj_props.object_offset[:2]
            a = obj_props.object_segment_a[:2]
            b = obj_props.object_segment_b[:2]
            size = (obj_props.object_width, obj_props.object_height)
            
            #TODO later
            vertices = []
            moment = 0
            if obj_props.object_moi_calc_type == 'FROM_BODY':
                if moment_type == 'CIRCLE':
                    moment = pymunk.moment_for_circle(mass, inner_radius, outer_radius, offset)
                elif moment_type == 'BOX':
                    moment = pymunk.moment_for_box(mass, size)
                elif moment_type == 'POLY':
                    #moment = pymunk.moment_for_poly(mass, vertices, offset, radius)
                    moment = obj_props.object_moi
                elif moment_type == 'SEGMENT':
                    moment = pymunk.moment_for_segment(mass, a, b, radius)
                elif moment_type == 'MANUAL':
                    moment = obj_props.object_moi
            else:
                #calculate moment FROM_SHAPE automatically
                pass
            
            body = pymunk.Body(mass, moment)
            if obj_props.object_type == 'DYNAMIC':
                body.body_type = pymunk.Body.DYNAMIC
            if obj_props.object_type == 'STATIC':
                body.body_type = pymunk.Body.STATIC
            if obj_props.object_type == 'KINEMATIC':
                body.body_type = pymunk.Body.KINEMATIC
            body.angle = obj.rotation_euler.z
            body.angular_velocity = obj_props.object_angular_velocity
            body.velocity = obj_props.object_velocity[:2]
            body.position = obj.location[:2]
            shape_props = obj.twod_shape_props
            
            for shape_prop in shape_props:
                shape = None
                radius = shape_prop.shape_radius
                offset = shape_prop.shape_offset[:2]
                mass = shape_prop.shape_mass
                density = shape_prop.shape_density
                surface_velocity = shape_prop.shape_surface_velocity[:2]
                elasticity = shape_prop.shape_elasticity
                friction = shape_prop.shape_friction
                start_point = shape_prop.shape_segment_a[:2]
                end_point = shape_prop.shape_segment_b[:2]
                
                if shape_prop.collision_shape == 'CIRCLE':
                    shape = pymunk.Circle(body, radius, offset)
                if shape_prop.collision_shape == 'POLY':
                    vertices = []
                    if shape_prop.poly_shape == 'CONVEX_HULL' and obj.type != 'GPENCIL':
                        vertices = self._get_vertices(obj)
                    elif shape_prop.shape_from_curve is not None:
                        vertices = self._get_vertices(shape_prop.shape_from_curve, curve=True)
                    shape = pymunk.Poly(body, vertices, radius=radius)
                if shape_prop.collision_shape == 'SEGMENT':
                    shape = pymunk.Segment(body, start_point, end_point, radius)
                
                if mass == 0.0:
                    shape.density = density
                else:
                    shape.mass = mass
                shape.surface_velocity = surface_velocity
                shape.elasticity = elasticity
                shape.friction = friction
                shape_prop.name = str(uuid.uuid4())
                self.space.add(shape)
                self.collision_shapes[shape_prop.name] = shape
                
            obj.twod_object_props.name = str(uuid.uuid4())
            self.rigid_bodies[obj_props.name] = body
            self.space.add(body)
            
            
    def clean_world(self):
        self.space = None
        
    
    def create_constraints(self, context):
        for obj in self.constraints:
            props = obj.twod_constraint_props
            anchor_a = props.anchor_point_a[:2]
            anchor_b = props.anchor_point_b[:2]
            constraint_type = props.constraint_type
            if props.object_a is None or props.object_b is None:
                continue
            
            body_a = self.rigid_bodies[props.object_a.twod_object_props.name]
            body_b = self.rigid_bodies[props.object_b.twod_object_props.name]
            
            if constraint_type == 'PIN':
                constraint = pymunk.PinJoint(body_a, body_b, anchor_a, anchor_b)
                if props.pin_use_distance:
                    constraint.distance = props.pin_joint_distance
            if constraint_type == 'SLIDE':
                constraint = pymunk.SlideJoint(body_a, body_b, anchor_a, anchor_b, props.joint_min, props.joint_max)
            if constraint_type == 'PIVOT':
                if props.single_pivot:
                    constraint = pymunk.PivotJoint(body_a, body_b, obj.location[:2])
                else:
                    constraint = pymunk.PivotJoint(body_a, body_b, anchor_a, anchor_b)
            if constraint_type == 'GROOVE':
                constraint = pymunk.GrooveJoint(body_a, body_b, props.groove_joint_a[:2], props.groove_joint_b[:2], anchor_b)
            if constraint_type == 'SPRING':
                constraint = pymunk.DampedSpring(body_a, body_b, anchor_a, anchor_b, props.damped_spring_rest_length, props.spring_stiffness, props.spring_damping)
                constraint.damping = props.spring_damping
            if constraint_type == 'ROTARY_SPRING':
                constraint = pymunk.DampedRotarySpring(body_a, body_b, props.damped_rotary_spring_rest_angle, props.spring_stiffness, props.spring_damping)
            if constraint_type == 'ROTARY_LIMIT':
                constraint = pymunk.RotaryLimitJoint(body_a, body_b, props.joint_min, props.joint_max)
            if constraint_type == 'RATCHET':
                constraint = pymunk.RatchetJoint(body_a, body_b, props.joint_phase , props.ratchet_joint_ratchet)
                constraint.angle = props.ratchet_joint_angle
            if constraint_type == 'GEAR':
                constraint = pymunk.GearJoint(body_a, body_b, props.joint_phase, props.gear_joint_ratio)
            if constraint_type == 'SIMPLE_MOTOR':
                constraint = pymunk.SimpleMotor(body_a, body_b, props.simple_motor_rate)
            
            constraint.max_bias = props.max_bias
            constraint.max_force = props.max_force
            constraint.error_bias = props.error_bias
            props.name = str(uuid.uuid4())
            self.constraints_dict[props.name] = constraint
            self.space.add(constraint)
                
        
        
        
    def _get_vertices(self, obj, curve=False):
        if curve:
            points = [x.co[:2] for x in obj.to_mesh().vertices]
        else:
            points = [x.co[:2] for x in obj.data.vertices]
        hull = ConvexHull(points)
        verts = []
        for x in hull.simplices:
            verts.append(tuple(x.coords[0]))
            verts.append(tuple(x.coords[1]))
        return verts
        
    def update(self, context, destroy):
        if destroy:
            self.create_world(context)
        else:
            self.update_bodies()
    
             
    
    def step(self):
        steps_per_sec = self.props.steps_per_sec
        dt = 1.0/steps_per_sec     
        self.space.step(dt)
        
    def update_bodies(self, context):
        props =  context.scene.twod_props
        if props.world_collection is None:
            objects = context.scene.objects
        else:
            objects = props.world_collection.objects
        for obj in objects:
            if obj.twod_object_props.object_enabled:
                self.bodies.add(obj)
            if obj.twod_constraint_props.constraint_enabled:
                self.constraints.add(obj)
    
    def update_animated_props(self):
        #update animation data if any
        
        for obj in self.bodies:
            if not obj.twod_object_props.object_is_animated:
                continue
            obj_props = obj.twod_object_props
            body_name = obj.twod_object_props.name
            body = self.rigid_bodies[body_name]
            body.angular_velocity = obj_props.object_angular_velocity
            body.velocity = obj_props.object_velocity[:2]
            body.position = obj.location[:2]
            body.angle = obj.rotation_euler.z
            shape_props = obj.twod_shape_props
            
            
            for shape_prop in shape_props:
                shape = self.collision_shapes[shape_prop.name]
                radius = shape_prop.shape_radius
                offset = shape_prop.shape_offset[:2]
                mass = shape_prop.shape_mass
                density = shape_prop.shape_density
                surface_velocity = shape_prop.shape_surface_velocity[:2]
                elasticity = shape_prop.shape_elasticity
                friction = shape_prop.shape_friction
                start_point = shape_prop.shape_segment_a[:2]
                end_point = shape_prop.shape_segment_b[:2]
                
                if shape_prop.collision_shape == 'CIRCLE':
                    shape.unsafe_set_offset(offset)
                if shape_prop.collision_shape == 'POLY':
                    shape.unsafe_set_radius(radius)
                if shape_prop.collision_shape == 'SEGMENT':
                    shape.a = start_point
                    shape.b = end_point
                    shape.unsafe_set_radius(radius)
                
                shape.surface_velocity = surface_velocity
                shape.elasticity = elasticity
                shape.friction = friction
        
        
    def update_objects(self, frame):    
        self.update_animated_props()         
        for obj in self.bodies:
            if obj.animation_data and obj.animation_data.action:
                if obj.animation_data.action.users > 1:
                    obj.animation_data_clear()
            if obj.twod_object_props.object_is_animated:
                continue
            body_name = obj.twod_object_props.name
            body = self.rigid_bodies[body_name]
            obj.location = (body.position.x, body.position.y, 0)
            obj.rotation_euler.z = body.angle
            obj.keyframe_insert('location', index=0, frame=frame)
            obj.keyframe_insert('location', index=1, frame=frame)
            obj.keyframe_insert('rotation_euler', index=2, frame=frame)
    
    def clear_keyframe(self):
        for obj in self.bodies:
            try:
                obj.keyframe_delete('location', index=0)
                obj.keyframe_delete('location', index=1)
                obj.keyframe_delete('rotation_euler', index=2)
            except:
                pass
        
class ClearBake(bpy.types.Operator):
    ''' Clear Baked Animation '''
    bl_idname = 'twod.clear_bake'
    bl_label = 'Clear Bake'
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        global world
        props = context.scene.twod_props
        world = None
        world = World(context)
        for frame in range(props.space_frame_end, props.space_frame_start-1, -1):
            context.scene.frame_set(frame)
            #context.scene.frame_current = frame
            world.clear_keyframe()
        return {'FINISHED'}

class BakePhysics(bpy.types.Operator):
    ''' Bake 2D Physics '''
    bl_idname = 'twod.bake_physics'
    bl_label = 'Bake 2D Physics'
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        #global world
        wm = context.window_manager
        props = context.scene.twod_props
        wm.progress_begin(0, 100)
        context.scene.frame_set(props.space_frame_start)
        #context.scene.frame_current = props.space_frame_start
        world = None
        world = World(context)
        for frame in range(props.space_frame_start, props.space_frame_end + 1):
            #context.scene.frame_current = frame
            context.scene.frame_set(frame)
            world.update_objects(frame)
            world.step()
            progress = (props.space_frame_end - frame)//(props.space_frame_end - props.space_frame_start)
            wm.progress_update(int(progress))
        wm.progress_end()
        del world.space
        world = None
        return {'FINISHED'}


class CreateConstraint(bpy.types.Operator):
    ''' Create Constraint '''
    bl_idname = 'twod.create_constraint'
    bl_label = 'Create Constraint'
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        world_exists = context.scene.twod_props.twod_world_exists
        return context.active_object is not None and world_exists
    
    def execute(self, context):
        constraint_props = context.object.twod_constraint_props
        constraint_props.constraint_enabled = True
        
        return {'FINISHED'}

class DeleteConstraint(bpy.types.Operator):
    ''' Delete Constraint '''
    bl_idname = 'twod.delete_constraint'
    bl_label = 'Delete Constraint'
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        constraint_props = context.object.twod_constraint_props
        constraint_props.constraint_enabled = False
        return {'FINISHED'}

import pickle
class SaveAsPickle(bpy.types.Operator):
    ''' Save as Pickle '''
    bl_idname = 'twod.save_as_pickle'
    bl_label = 'Save as Pickle'
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        global world
        world = World(context)
        
        with open('space.pickle', 'wb') as handle:
            pickle.dump(world.space, handle, protocol=pickle.HIGHEST_PROTOCOL)
        world = None
        return {'FINISHED'}

class AddCollisionShape(bpy.types.Operator):
    ''' Add Collision Shape '''
    bl_idname = 'twod.add_collision_shape'
    bl_label = 'Add 2D Collision shape'
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context):
        return context.active_object is not None
    
    def execute(self, context): 
        shape_props = context.object.twod_shape_props
        shape = shape_props.add()
        shape.name = str(uuid.uuid4())
        return {'FINISHED'}
        
        
class RemoveCollisionShape(bpy.types.Operator):
    ''' Remove Collision Shape '''
    bl_idname = 'twod.remove_collision_shape'
    bl_label = 'Remove 2D Collision shape'
    bl_options = {'REGISTER', 'UNDO'}
    
    name: bpy.props.StringProperty()
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None
    def execute(self, context):
        shape_props = context.object.twod_shape_props
        if len(shape_props) == 1:
            self.report({'WARNING'}, '2D physics object should have atleast one collision shape attached')
        else:
            for i,prop in enumerate(shape_props):
                if prop.name == self.name:
                    shape_props.remove(i)
                    break
        
        #world.update(context, True)
        return {'FINISHED'}
        

class CreateTwoDObject(bpy.types.Operator):
    ''' Create a 2D object '''
    bl_idname = 'twod.create_object'
    bl_label = 'Create 2D Object'
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        world_exists = context.scene.twod_props.twod_world_exists
        return context.active_object is not None and world_exists
    
    def execute(self, context):
        obj_props = context.object.twod_object_props
        obj_props.object_enabled = True
        obj_props.name = str(uuid.uuid4())
        #Create all details
        bpy.ops.twod.add_collision_shape()
        return {'FINISHED'}
    
class DeleteTwoDObject(bpy.types.Operator):
    ''' Delete 2D Object '''
    bl_idname = 'twod.delete_object'
    bl_label = 'Delete 2D Object'
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None
    
    def execute(self, context):
        obj_props = context.object.twod_object_props
        shape_props = context.object.twod_shape_props
        obj_props.object_enabled = False
        
        #remove object props
        shape_props = context.object.twod_shape_props
        shape_props.clear()
        return {'FINISHED'}

class CreateTwoDSpace(bpy.types.Operator):
    """Create a 2D physics space"""
    bl_idname = "twod.create_space"
    bl_label = "Create 2D Space"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.twod_props
        props.twod_world_exists = True
        #global world
        #world = World(context)
        return {'FINISHED'}

class DeleteTwoDSpace(bpy.types.Operator):
    ''' Delete 2D physics space '''
    bl_idname = 'twod.delete_space'
    bl_label = 'Delete 2D Space'
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.twod_props
        props.twod_world_exists = False
        props.debug_visualisation = False
        
        # remove all the contents of the space here
        for obj in context.scene.objects:
            obj.twod_object_props.object_enabled = False
            obj.twod_shape_props.clear()
        
        try:
            del world.space
        except:
            pass
        world = None
        return {'FINISHED'}


class RefreshSpace(bpy.types.Operator):
    ''' Refresh 2D physics space '''
    bl_idname = 'twod.refresh_space'
    bl_label = 'Refresh 2D Space'
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        global world
        world = None
        world = World(context)
        return {'FINISHED'}


''' All drawing functions goes here '''

shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')


def get_circle_coords(pos, radius, offset, angle):
    #color=(0,0,0.7,1), center=(0,0), radius=1, segments=32, line_width=1):
    center = Vec2d(pos) + Vec2d(offset).rotated(angle)
    segments = int(12 * math.sqrt(radius))
    if segments <= 12:
        segments = 12
    theta = 2 * 3.1415926 / segments
    c = math.cos(theta) 
    s = math.sin(theta)
    x = radius
    y = 0
    
    coords = []
    first_vert = None
    for i in range (segments):
        point = Vec2d((x,y)) + center
        if i != 0:
            coords.append(point)
        else:
            first_vert = point
        coords.append(point) 
        t = x
        x = c * x - s * y
        y = s * t + c * y
    coords.append(first_vert)
    return coords

def draw_polygon(verts, radius, outline_color, fill_color):
    bgl.glLineWidth(radius)
    coords = []
    for vert in verts:
        coords.append((vert[0], vert[1], 0.0))
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    shader.bind()
    shader.uniform_float("color", outline_color)
    #self.batches.append(batch)
    batch.draw(shader)
    bgl.glLineWidth(1)


from pyhull.convex_hull import ConvexHull
def get_convex_hull(context):
    points = [x.co[:2] for x in context.object.data.vertices]
    hull = ConvexHull(points)
    verts = []
    for x in hull.simplices:
        verts.append(Vec2d(x.coords[0].tolist()))
        verts.append(Vec2d(x.coords[1].tolist()))
    return verts

def get_vertices_from_curve(obj):
    points = [x.co[:2] for x in obj.to_mesh().vertices]
    hull = ConvexHull(points)
    verts = []
    for x in hull.simplices:
        verts.append(Vec2d(x.coords[0].tolist()))
        verts.append(Vec2d(x.coords[1].tolist()))
    return verts

def get_vertices_from_gpencil(obj, frame):
    pass


def get_shapes_to_draw(context):
    shapes = []
    if context.object is not None and context.object.select_get():
        obj_props = context.object.twod_object_props
        shape_props = context.object.twod_shape_props
        for shape_prop in shape_props:
            position = Vec2d(context.object.location[:2])
            angle = context.object.rotation_euler.z
            radius = shape_prop.shape_radius
            offset = shape_prop.shape_offset[:2]
            a = Vec2d(shape_prop.shape_segment_a[:2])
            b = Vec2d(shape_prop.shape_segment_b[:2])
            collision_shape = shape_prop.collision_shape
            vertices = []
            if shape_prop.poly_shape == 'CONVEX_HULL' and context.object.type != 'GPENCIL':
                vertices = get_convex_hull(context)
            elif shape_prop.shape_from_curve is not None:
                #From Curve
                vertices = get_vertices_from_curve(shape_prop.shape_from_curve)
            shapes.append({'position': position, 'angle': angle, 'radius': radius, 'a': a, 'b': b, 'collision_shape': collision_shape, 'vertices': vertices, 'offset': offset})
    return shapes    

def draw_callback_3d(self, context):
    #global world
    #if world is None:
    #    world = World(context)
    ### Draw space
    #self.space.debug_draw(self.draw_options)
    bgl.glEnable(bgl.GL_BLEND)
    verts = []
    shapes = get_shapes_to_draw(context)
    for shape in shapes:
        if shape['collision_shape'] == 'POLY':
            new_shape = True
            first_vert = None
            for v in shape['vertices']:
                v_r = v.rotated(shape['angle'])
                x = v_r[0] + shape['position'][0]
                y = v_r[1] + shape['position'][1]
                verts.append((x, y))
        if shape['collision_shape'] == 'CIRCLE':
            verts += get_circle_coords(shape['position'], shape['radius'], shape['offset'], shape['angle'])
        if shape['collision_shape'] == 'SEGMENT':
            pv1 = shape['offset'] + shape['position'] + shape['a'].rotated(shape['angle'])
            pv2 = shape['offset'] + shape['position'] + shape['b'].rotated(shape['angle'])
            verts.append(pv1)
            verts.append(pv2)
    draw_polygon(verts, 1, (1,1,1,1), (1,1,1,1))
    # Restore defaults
    bgl.glLineWidth(1)
    bgl.glDisable(bgl.GL_BLEND)

'''
def draw_world(context):
    props = context.scene.twod_props
    draw_rectangle(color=COLOR_WORLD, width=props.world_width, height=props.world_height)
'''    
    
_handle_3d = None

''' Toggle between debug visualisation '''
def debug_vis_update(self,context):
    global _handle_3d
    args = (self, context)
    if self.debug_visualisation :
        _handle_3d = bpy.types.SpaceView3D.draw_handler_add(draw_callback_3d, args, 'WINDOW', 'POST_VIEW')
    else:
        bpy.types.SpaceView3D.draw_handler_remove(_handle_3d, 'WINDOW')
        _handle_3d = None

def delete_collision_shape(self, context):
	bpy.ops.twod.remove_collision_shape(name=self.name)





moi_enum_types = [
    ("BOX", "Box", "", 1),
    ("CIRCLE", "Circle", "", 2),
    ("POLY", "Poly", "", 3),
    ("SEGMENT", "Segment", "", 4),
    ("MANUAL", "Manual", '', 5)
]

moi_calc_types = [
    ("FROM_SHAPE", "From Shape", "", 2),
    ("FROM_BODY", "From Body", "", 1)
]

poly_shape_types = [
    ("CONVEX_HULL", "Convex Hull", "", 1),
    ("CURVE", "From Curve", "", 2),
]

shape_enum_types = [
    ("CIRCLE", "Circle", "", 1),
    ("POLY", "Poly", "", 2),
    ("SEGMENT", "Segment", "", 3),
]

object_types = [
    ("DYNAMIC", "Dynamic", "", 1),
    ("STATIC", "Static", "", 2),
    ("KINEMATIC", "Kinematic", "", 3),
]

constraint_types = [
    ("PIN", "Pin Joint", "", 1),
    ("PIVOT", "Pivot Joint", "", 2),
    ("SLIDE", "Slide Joint", "", 3),
    ("GROOVE", "Groove Joint", "", 4),
    ("SPRING", "Spring", "", 5),
    ("ROTARY_SPRING", "Rotary Spring", "", 6),
    ("ROTARY_LIMIT", "Rotary Limit Joint", "", 7),
    ("RATCHET", "Ratchet Joint", "", 8),
    ("GEAR", "Gear Joint", "", 9),
    ("SIMPLE_MOTOR", "Simple Motor", "", 10)
]

def scene_updated(scene):
    ob = bpy.context.object
    if ob is not None:
        pass


def collision_shape_update(self, context):
    if self.collision_shape == 'CIRCLE':
        self.shape_radius = 1
    else:
        self.shape_radius = 0


def poll_for_curves(self, object):
    return object.type == 'CURVE'

def poll_for_rigidbodies(self, object):
    return object.twod_object_props.object_enabled

from math import inf
class TwoDCollisionShapeProperties(PropertyGroup):
    delete_shape: bpy.props.BoolProperty(name='Delete Shape', default=False, update=delete_collision_shape)
    collision_shape: bpy.props.EnumProperty(name='Collision Shape', items=shape_enum_types, update=collision_shape_update)
    shape_density : bpy.props.FloatProperty(name='Density', default=1.0, options={'TEXTEDIT_UPDATE'}, description='The density of this shape.')
    shape_mass : bpy.props.FloatProperty(name='Mass', default=1.0, options={'TEXTEDIT_UPDATE'}, description='The mass of this shape.')
    shape_friction : bpy.props.FloatProperty(name='Friction', default=0.7, min=0.0, max=2.0, description='Friction coefficient.')
    shape_radius : bpy.props.FloatProperty(name='Radius', default=1.0, min=0.0, description='The Radius')
    shape_elasticity : bpy.props.FloatProperty(name='Elasticity', default=0.5, min=0.0, max=1.0, description='Elasticity of the shape.')
    shape_offset: bpy.props.FloatVectorProperty(name='Offset', default=(0,0,0))
    shape_segment_a: bpy.props.FloatVectorProperty(name='Start Point', default=(0,0,0), description='The first endpoint of the segment')
    shape_segment_b: bpy.props.FloatVectorProperty(name='End Point', default=(0,0,0), description='The second endpoint of the segment')
    shape_surface_velocity: bpy.props.FloatVectorProperty(name='Surface Velocity', default=(0,0,0),description='The surface velocity of the object. This value is only used when calculating friction, not resolving the collision.')
    show_panel: bpy.props.BoolProperty(name='Collision Shape', default=False)
    shape_from_curve: bpy.props.PointerProperty(name='From Curve', type=bpy.types.Object, poll=poll_for_curves, description='Get collision shape from curve.')
    poly_shape: bpy.props.EnumProperty(name='Shape From', items=poly_shape_types)



class TwoDProperties(PropertyGroup):
    debug_visualisation : bpy.props.BoolProperty(name="Enable Visualisation", update=debug_vis_update, options={'TEXTEDIT_UPDATE'}, description='Show collision hit box of the selected object')
    world_height : bpy.props.FloatProperty(name="Height", default=100.0, options={'TEXTEDIT_UPDATE'})
    world_width : bpy.props.FloatProperty(name="Width", default=100.0, options={'TEXTEDIT_UPDATE'})
    show_world_panel: bpy.props.BoolProperty(name='World Settings', default=False, options={'TEXTEDIT_UPDATE'})
    twod_world_exists: bpy.props.BoolProperty(name='Create 2D World', default=False, options={'TEXTEDIT_UPDATE'}, description='Spaces are the basic unit of simulation. You add rigid bodies, shapes and joints to it and then step them all forward together through time.')
    world_gravity: bpy.props.FloatVectorProperty(name='Gravity', default=(0.0, -90.0, 0.0), description='Global gravity applied to the space.')
    collision_persistence: bpy.props.IntProperty(name="Collision Persistence", default=3, min=0, description='The number of frames the space keeps collision solutions around for. Helps prevent jittering contacts from getting worse. This defaults to 3.')
    collision_bias: bpy.props.FloatProperty(name="Collision Bias", default=math.pow(1.0-0.1, 60.0), min=0.0, max=1.0, description='Determines how fast overlapping shapes are pushed apart. To improve stability, set this as high as you can without noticeable overlapping. ')
    collision_slop: bpy.props.FloatProperty(name="Collision Slop", default=0.1, min=0.0,description='Amount of overlap between shapes that is allowed.')
    world_damping: bpy.props.FloatProperty(name='Damping', default=1, min=0.0, max=1.0, description='Amount of simple damping to apply to the space.')
    iterations: bpy.props.IntProperty(name='Iterations', default=10, min=1, options={'TEXTEDIT_UPDATE'}, description='Iterations allow you to control the accuracy of the solver.')
    steps_per_sec: bpy.props.IntProperty(name='Steps Per Second', default=60,min=1, options={'TEXTEDIT_UPDATE'}, description='No of steps to simulate in a second, higher is better but slower.')
    space_frame_start: bpy.props.IntProperty(name='Start Frame', default=1, min=0, options={'TEXTEDIT_UPDATE'}, description='Start frame of simulation')
    space_frame_end: bpy.props.IntProperty(name='End Frame', default=100, min=0, options={'TEXTEDIT_UPDATE'}, description='End frame of simulation')

    world_collection: bpy.props.PointerProperty(name='World Collection', type=bpy.types.Collection)

class TwoDConstraintProperties(PropertyGroup):
    anchor_point_a: bpy.props.FloatVectorProperty(name='Anchor Point A', default=(0,0,0))
    anchor_point_b: bpy.props.FloatVectorProperty(name='Anchor Point B', default=(0,0,0))
    object_a: bpy.props.PointerProperty(name='First Object', type=bpy.types.Object, poll=poll_for_rigidbodies, description='The first of the two bodies constrained')
    object_b: bpy.props.PointerProperty(name='Second Object', type=bpy.types.Object, poll=poll_for_rigidbodies, description='The second of the two bodies constrained')
    collide_bodies: bpy.props.BoolProperty(name='Collide Bodies', default=True, description=' Ignores the collisions if this property is set to False on any constraint that connects the two bodies.')
    error_bias: bpy.props.FloatProperty(name='Error Bias', default= math.pow(1.0 - 0.1, 60.0), description='The percentage of joint error that remains unfixed after a second.')
    max_bias: bpy.props.FloatProperty(name='Max Bias', default=inf, description='The maximum speed at which the constraint can apply error correction.')
    max_force: bpy.props.FloatProperty(name='Max Force', default=inf, description='The maximum force that the constraint can use to act on the two bodies.')
    pin_joint_distance: bpy.props.FloatProperty(name='Distance', default=0)
    joint_min: bpy.props.FloatProperty(name='Min', default=0)
    joint_max: bpy.props.FloatProperty(name='Max', default=1)
    single_pivot: bpy.props.BoolProperty(name='Use empty as pivot', default=False)
    pivot_joint_pivot: bpy.props.FloatVectorProperty(name='Pivot', default=(0,0,0))
    groove_joint_a: bpy.props.FloatVectorProperty(name='First Groove', default=(0,0,0))
    groove_joint_b: bpy.props.FloatVectorProperty(name='Second Groove', default=(0,0,0))
    damped_spring_rest_length: bpy.props.FloatProperty(name='Rest Length', default=0, description='The distance the spring wants to be.')
    spring_stiffness: bpy.props.FloatProperty(name='Stiffness', default=0, description='The spring constant (Young’s modulus).')
    spring_damping: bpy.props.FloatProperty(name='Damping', default=0)
    damped_rotary_spring_rest_angle: bpy.props.FloatProperty(name='Rest Angle', default=0, description='The relative angle in radians that the bodies want to have.')
    ratchet_joint_angle: bpy.props.FloatProperty(name='Angle', default=0)
    ratchet_joint_ratchet: bpy.props.FloatProperty(name='Ratchet', default=0)
    joint_phase: bpy.props.FloatProperty(name='Phase', default=0)
    gear_joint_ratio: bpy.props.FloatProperty(name='Ratio', default=0)
    simple_motor_rate: bpy.props.FloatProperty(name='Rate', default=0, description='The desired relative angular velocity')
    constraint_type: bpy.props.EnumProperty(name='Type', items=constraint_types)
    constraint_enabled: bpy.props.BoolProperty(name='Constraint Enabled', default=False)
    pin_use_distance: bpy.props.BoolProperty(name='Use Distance', default=False)
    

#find rest of the tunable properties and add progressively
class TwoDObjectProperties(PropertyGroup):
    object_mass : bpy.props.FloatProperty(name='Mass', default=1.0, min=0.0, options={'TEXTEDIT_UPDATE'}, description='Mass of the body.')
    object_radius : bpy.props.FloatProperty(name='Radius', default=1.0, min=0.0)
    object_moi : bpy.props.FloatProperty(name='Moment', default=1.0, min=0.0, description='Moment of inertia of the body.')
    object_moi_calc_type : bpy.props.EnumProperty(name='Moment Calculation', items=moi_calc_types)
    object_enabled: bpy.props.BoolProperty(name='Object Enabled', default=False)
    object_moi_type : bpy.props.EnumProperty(name='Moment', items=moi_enum_types)
    object_segment_a: bpy.props.FloatVectorProperty(name='Start Point', default=(0,0,0))
    object_segment_b: bpy.props.FloatVectorProperty(name='End Point', default=(10,0,0))
    object_inner_radius: bpy.props.FloatProperty(name='Inner Radius', default=0.0, min=0.0)
    object_outer_radius: bpy.props.FloatProperty(name='Outer Radius', default=1.0, min=0.0)
    object_offset: bpy.props.FloatVectorProperty(name='Offset', default=(0,0,0))
    object_width: bpy.props.FloatProperty(name='Width', default=1.0, min=0.0)
    object_height: bpy.props.FloatProperty(name='Height', default=1.0, min=0.0)
    object_type : bpy.props.EnumProperty(name='Body Type', items=object_types)
    object_angle: bpy.props.FloatProperty(name='Angle', default=0.0, description='Rotation of the body in radians.')
    object_velocity: bpy.props.FloatVectorProperty(name='Velocity', default=(0,0,0), description='The velocity of the body.')
    object_angular_velocity: bpy.props.FloatProperty(name='Angular Velocity', default=0.0, description='The angular velocity of the body in radians per second.')
    object_is_animated: bpy.props.BoolProperty(name='Animated', default=False, description='Rigidbody is animated. i.e. take locations from the animation.')

''' Draw panel here '''

class TwoDPhysicsConstraints(Panel):
    bl_label = '2D Physics Constraint'
    bl_idname = 'TWOD_PT_constraint'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'physics'
    
    @classmethod
    def poll(self, context):
        return context.object.type in ['EMPTY', 'MESH', 'GPENCIL']
    
    def draw(self, context):
        layout = self.layout
        props = context.object.twod_constraint_props
    
        if props.constraint_enabled:
            
            row = layout.row()
            row.prop(props, 'constraint_type')
            
            split = layout.split()
            
            col = split.column(align=True)
            col.label(text='First Object')
            col.label(text='Second Object')
            
            
            
            col = split.column(align=True)
            col.prop(props, 'object_a', text='')
            col.prop(props, 'object_b', text='')
            
            if props.constraint_type in ['PIN', 'SPRING', 'SLIDE']:
                col = layout.column(align=True)
                col.label(text='Anchor Point A')
                row = col.row(align=True)
                row.prop(props, 'anchor_point_a', index=0,text='X')
                row.prop(props, 'anchor_point_a', index=1,text='Y')
                
                col = layout.column(align=True)
                col.label(text='Anchor Point B')
                row = col.row(align=True)
                row.prop(props, 'anchor_point_b', index=0,text='X')
                row.prop(props, 'anchor_point_b', index=1,text='Y')

            row = layout.row()
            row.prop(props, 'collide_bodies')
            
            row = layout.row(align=True)
            if props.constraint_type == 'PIN':
                row.prop(props, 'pin_use_distance')
                if props.pin_use_distance:
                    row.prop(props, 'pin_joint_distance')
            if props.constraint_type == 'PIVOT':
                row.prop(props, 'single_pivot')
                if not props.single_pivot:
                    col = layout.column(align=True)
                    col.label(text='Anchor Point A')
                    row = col.row(align=True)
                    row.prop(props, 'anchor_point_a', index=0,text='X')
                    row.prop(props, 'anchor_point_a', index=1,text='Y')
                    
                    col = layout.column(align=True)
                    col.label(text='Anchor Point B')
                    row = col.row(align=True)
                    row.prop(props, 'anchor_point_b', index=0,text='X')
                    row.prop(props, 'anchor_point_b', index=1,text='Y')
            if props.constraint_type == 'SLIDE':
                row.prop(props, 'joint_min')
                row.prop(props, 'joint_max')
            if props.constraint_type == 'GROOVE':
                row.label(text='Groove A')
                subrow = row.row(align=True)
                subrow.prop(props,'groove_joint_a', index=0, text='X')
                subrow.prop(props,'groove_joint_a', index=1, text='Y')
                row = layout.row(align=True)
                row.label(text='Groove B')
                subrow = row.row(align=True)
                subrow.prop(props,'groove_joint_b', index=0, text='X')
                subrow.prop(props,'groove_joint_b', index=1, text='Y')
                
                col = layout.column(align=True)
                col.label(text='Anchor Point B')
                row = col.row(align=True)
                row.prop(props, 'anchor_point_b', index=0,text='X')
                row.prop(props, 'anchor_point_b', index=1,text='Y')
            if props.constraint_type == 'SPRING':
                row.prop(props, 'damped_spring_rest_length')
                row = layout.row(align=True)
                row.prop(props, 'spring_stiffness')
                row.prop(props, 'spring_damping')
                
            if props.constraint_type == 'ROTARY_SPRING':
                row.prop(props, 'damped_rotary_spring_rest_angle')
                row = layout.row(align=True)            
                row.prop(props, 'spring_stiffness')
                row.prop(props, 'spring_damping')
            if props.constraint_type == 'ROTARY_LIMIT':
                row.prop(props, 'joint_min')
                row.prop(props, 'joint_max')
            if props.constraint_type == 'RATCHET':
                row.prop(props, 'ratchet_joint_angle')
                row = layout.row(align=True)            
                row.prop(props, 'ratchet_joint_ratchet')
                row.prop(props, 'joint_phase')
            if props.constraint_type == 'GEAR':
                row.prop(props, 'joint_phase')
                row.prop(props, 'gear_joint_ratio')
            if props.constraint_type == 'SIMPLE_MOTOR':
                row.prop(props, 'simple_motor_rate')
            
            row = layout.row(align=True)
            row.prop(props, 'error_bias')
            row.prop(props, 'max_bias')
            
            row = layout.row(align=True)
            row.prop(props, 'max_force')
            
            row = layout.row()
            row.operator('twod.delete_constraint')
        else:
            row = layout.row()
            row.operator('twod.create_constraint')
        
class TwoDPhysicsObject(Panel):
    bl_label = '2D Physics Object'
    bl_idname = 'TWOD_PT_object'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'physics'
    
    @classmethod
    def poll(cls, context):
        return context.object.type == 'GPENCIL' or context.object.type == 'MESH'
    
    def draw(self,context):
        layout = self.layout
        scene = context.scene
        scene_props = scene.twod_props
        obj_props = context.object.twod_object_props
        row = layout.row()
        
        if obj_props.object_enabled:
            row.operator('twod.delete_object', icon='PANEL_CLOSE')
        else:
            row.operator('twod.create_object')
        
        if obj_props.object_enabled:
            row = layout.row(align=True)
            row.prop(obj_props, 'object_mass')
            row = layout.row()
            row.prop(obj_props, 'object_type', expand=True)
            
            #row = layout.row()
            #row.prop(obj_props, 'object_angle')
            row = layout.row()
            row.prop(obj_props, 'object_is_animated')
            
            row = layout.row(align=True)
            row.label(text='Velocity')
            row.prop(obj_props, 'object_velocity', index=0, text='X')
            row.prop(obj_props, 'object_velocity', index=1, text='Y')
            
            row = layout.row(align=True)
            row.prop(obj_props, 'object_angular_velocity')
            
            col = layout.column(align=True)
            col.label(text='Calculate Moment')
            row = col.row(align=True)
            row.prop(obj_props, 'object_moi_calc_type', expand=True)
            if obj_props.object_moi_calc_type == 'FROM_BODY':
                row = layout.row()
                row.prop(obj_props, 'object_moi_type', expand=True) 
                row = layout.row(align=True)
                if obj_props.object_moi_type == 'CIRCLE': 
                    row.prop(obj_props, 'object_inner_radius')
                    row.prop(obj_props, 'object_outer_radius')
                    row = layout.row(align=True)
                    row.prop(obj_props, 'object_segment_a', index=0, text='Offset X')
                    row.prop(obj_props, 'object_segment_a', index=1, text='Offset Y')
                if obj_props.object_moi_type == 'MANUAL':
                    row.prop(obj_props, 'object_moi')
                if obj_props.object_moi_type == 'BOX':
                    row.prop(obj_props, 'object_width')
                    row.prop(obj_props, 'object_height')
                if obj_props.object_moi_type == 'SEGMENT':
                    col = layout.column(align=True)
                    row = col.row()
                    row.prop(obj_props, 'object_radius')
                    row = col.row()
                    row.label(text='Start Point')
                    row.label(text='End Point')
                    row = col.row()
                    col = row.column(align=True)
                    col.prop(obj_props, 'object_segment_a', index=0, text='X')
                    col.prop(obj_props, 'object_segment_a', index=1, text='Y')
                    col = row.column(align=True)
                    col.prop(obj_props, 'object_segment_b', index=0, text='X')
                    col.prop(obj_props, 'object_segment_b', index=1, text='Y')
                if obj_props.object_moi_type == 'POLY':
                    #show manual temporarily
                    row.prop(obj_props, 'object_moi')
            else:
                pass    
            
            shape_props = context.object.twod_shape_props

            for shape in shape_props:
                box = layout.box()
                row = box.row()
                if shape.show_panel:
                    row.prop(shape, "show_panel", icon="DOWNARROW_HLT", emboss=False)
                else:
                    row.prop(shape, "show_panel", icon="RIGHTARROW", emboss=False)
                    prop = row.prop(shape, 'delete_shape', icon='CANCEL')
                if shape.show_panel:
                    row = box.row()
                    row.prop(shape, 'collision_shape', expand=True)
                    row = box.row(align=True)
                    
                    if shape.collision_shape == 'POLY':
                        if context.object.type == 'GPENCIL':
                            shape.poly_shape == 'CURVE'
                            row = box.row()
                            row.prop(shape, 'shape_from_curve')
                        else:
                            row.prop(shape, 'poly_shape', expand=True)
                            if shape.poly_shape == 'CURVE':
                                row = box.row()
                                row.prop(shape, 'shape_from_curve')
                    
                    row = box.row(align=True)
                    #either mass or density, now just put both
                    row.prop(shape, 'shape_mass', expand=True)
                    row.prop(shape, 'shape_density', expand=True)
                    row = box.row(align=True)
                    row.prop(shape, 'shape_radius')
                    
                    if shape.collision_shape == 'SEGMENT':
                        col = box.column(align=True)
                        row = col.row()
                        row.label(text='Start Point')
                        row.label(text='End Point')
                        row = col.row()
                        col = row.column(align=True)
                        col.prop(shape, 'shape_segment_a', index=0, text='X')
                        col.prop(shape, 'shape_segment_a', index=1, text='Y')
                        col = row.column(align=True)
                        col.prop(shape, 'shape_segment_b', index=0, text='X')
                        col.prop(shape, 'shape_segment_b', index=1, text='Y')
                    
                    
                    row = box.row(align=True)
                    row.label(text='Offset')
                    row.prop(shape, 'shape_offset', index=0, text='X')
                    row.prop(shape, 'shape_offset', index=1, text='Y')
                    
                    
                    row = box.row(align=True)
                    row.label(text='Surface Velocity')
                    row.prop(shape, 'shape_surface_velocity', index=0, text='X')
                    row.prop(shape, 'shape_surface_velocity', index=1, text='Y')
                    
                    row = box.row(align=True)
                    row.prop(shape, 'shape_elasticity')
                    row.prop(shape, 'shape_friction')
                    
                    split = box.split()
                    row = split.row()
                    prop = row.prop(shape, 'delete_shape', icon='PANEL_CLOSE')
                    
                    if len(shape_props) == 1:
                        row.enabled = False
                    row = split.row(align=True)
                    prop = row.operator('twod.add_collision_shape', text='Add Shape')
    
class TwoDPhysicsWorld(Panel):
    """2D Physics"""
    bl_label = "2D Physics World"
    bl_idname = "TWOD_PT_world"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.twod_props
        
        row = layout.row()
        if props.twod_world_exists:
            row.operator('twod.delete_space', icon='PANEL_CLOSE')
        else:
            row.operator('twod.create_space')
        if props.twod_world_exists:
            box = layout.box()
            row = box.row()
            if props.show_world_panel:
                row.prop(props, "show_world_panel", icon="DOWNARROW_HLT", text="", emboss=False)
            else:
                row.prop(props, "show_world_panel", icon="RIGHTARROW", text="", emboss=False)
            
            row.label(text='World Settings')
            if props.show_world_panel:
                row = box.row()
                row.prop(props, 'debug_visualisation')
                #row = box.row(align=True)
                #row.prop(props, 'world_height')
                #row.prop(props, 'world_width')
                row = box.row()
                row.prop(props, 'world_collection')
                row = box.row(align = True)
                row.label(text = 'Gravity')
                
                row.prop(props, 'world_gravity', index=0, text='x')
                row.prop(props, 'world_gravity', index=1, text='y')
                col = box.column(align=True)
                col.prop(props, 'iterations')
                col.prop(props, 'steps_per_sec')
                
                col = box.column(align=True)     
                col.prop(props, 'collision_bias')
                col.prop(props, 'collision_persistence')
                col.prop(props, 'collision_slop')
                
                row = box.row()
                row.prop(props, 'world_damping')
                
                row = box.row(align=True)
                row.prop(props, 'space_frame_start')
                row.prop(props, 'space_frame_end')
                
                row = box.row()
                row.operator('twod.bake_physics')
                
                row = layout.row()
                row.operator('twod.clear_bake')
                row = layout.row()
                #row.operator('twod.refresh_space')
                
                row = layout.row()
                #row.operator('twod.save_as_pickle')


def step_simulation(scene):
    world.step()                
            

classes = [ClearBake, CreateConstraint, DeleteConstraint, TwoDPhysicsConstraints, TwoDConstraintProperties, SaveAsPickle, BakePhysics, RefreshSpace, TwoDProperties, TwoDObjectProperties, TwoDCollisionShapeProperties, DeleteTwoDSpace, DeleteTwoDObject, TwoDPhysicsWorld, TwoDPhysicsObject, CreateTwoDObject, CreateTwoDSpace, AddCollisionShape, RemoveCollisionShape]
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.twod_props = PointerProperty(
        type=TwoDProperties
    )
    bpy.types.Object.twod_object_props = PointerProperty(
        type=TwoDObjectProperties
    )
    bpy.types.Object.twod_shape_props = CollectionProperty(
        type=TwoDCollisionShapeProperties
    )
    bpy.types.Object.twod_constraint_props = PointerProperty(
        type=TwoDConstraintProperties
    )

def unregister():
    #del bpy.types.WindowManager.twod_props
    if _handle_3d is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle_3d, 'WINDOW')
    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
