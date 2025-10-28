use alum::{
    Adaptor, CrossProductAdaptor, DotProductAdaptor, FloatScalarAdaptor, Handle, HasIterators,
    HasTopology, PolyMeshT, VectorLengthAdaptor, VectorNormalizeAdaptor,
};
use three_d::{
    Context, CpuMaterial, CpuMesh, Cull, Gm, Indices, InnerSpace, InstancedMesh, Instances, Mat4,
    Mesh, Object, PhysicalMaterial, Positions, Quat, Srgba, Vec3, vec3,
};

pub struct MeshAdaptor;

impl Adaptor<3> for MeshAdaptor {
    type Vector = Vec3;

    type Scalar = f32;

    fn vector(coords: [Self::Scalar; 3]) -> Self::Vector {
        three_d::vec3(coords[0], coords[1], coords[2])
    }

    fn zero_vector() -> Self::Vector {
        three_d::vec3(0.0, 0.0, 0.0)
    }

    fn vector_coord(v: &Self::Vector, i: usize) -> Self::Scalar {
        v[i]
    }
}

impl FloatScalarAdaptor<3> for MeshAdaptor {
    fn scalarf32(val: f32) -> Self::Scalar {
        val
    }

    fn scalarf64(val: f64) -> Self::Scalar {
        val as f32
    }
}

impl CrossProductAdaptor for MeshAdaptor {
    fn cross_product(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        a.cross(b)
    }
}

impl VectorNormalizeAdaptor<3> for MeshAdaptor {
    fn normalized_vec(v: Self::Vector) -> Self::Vector {
        v.normalize()
    }
}

impl VectorLengthAdaptor<3> for MeshAdaptor {
    fn vector_length(v: Self::Vector) -> Self::Scalar {
        v.magnitude()
    }
}

impl DotProductAdaptor<3> for MeshAdaptor {
    fn dot_product(a: Self::Vector, b: Self::Vector) -> Self::Scalar {
        a.dot(b)
    }
}

pub type PolygonMesh = PolyMeshT<3, MeshAdaptor>;

pub struct MeshView {
    faces: Gm<Mesh, PhysicalMaterial>,
    vertices: Gm<InstancedMesh, PhysicalMaterial>,
    edges: Gm<InstancedMesh, PhysicalMaterial>,
}

impl MeshView {
    pub fn as_iter(&self) -> impl Iterator<Item = &dyn Object> {
        self.faces
            .into_iter()
            .chain(&self.vertices)
            .chain(&self.edges)
    }
}

fn mesh_view(
    mut mesh: PolygonMesh,
    context: &Context,
    vertex_radius: f32,
    edge_radius: f32,
    face_material: PhysicalMaterial,
    wireframe_material: PhysicalMaterial,
) -> Option<MeshView> {
    mesh.delete_isolated_vertices()
        .expect("Cannot delete isolated vertices");
    mesh.garbage_collection()
        .expect("Cannot garbage collect mesh");
    mesh.update_vertex_normals_accurate().unwrap();
    if mesh.num_faces() == 0 {
        return None;
    }
    let points = mesh.points();
    let points = points.try_borrow().expect("Cannot borrow points");
    let vnormals = mesh.vertex_normals().unwrap();
    let vnormals = vnormals.try_borrow().unwrap();
    let fstatus = mesh.face_status_prop();
    let fstatus = fstatus.try_borrow().unwrap();
    let cpumesh = CpuMesh {
        positions: Positions::F32(points.iter().map(|p| vec3(p.x, p.y, p.z)).collect()),
        indices: Indices::U32(
            mesh.triangulated_vertices(&fstatus)
                .flatten()
                .map(|v| v.index())
                .collect(),
        ),
        normals: Some(vnormals.to_vec()),
        ..Default::default()
    };
    let mut sphere = CpuMesh::sphere(8);
    sphere.transform(Mat4::from_scale(vertex_radius)).unwrap();
    let mut cylinder = CpuMesh::cylinder(10);
    cylinder
        .transform(Mat4::from_nonuniform_scale(1.0, edge_radius, edge_radius))
        .unwrap();
    Some(MeshView {
        faces: Gm::new(Mesh::new(context, &cpumesh), face_material),
        vertices: Gm::new(
            InstancedMesh::new(
                context,
                &Instances {
                    transformations: points
                        .iter()
                        .map(|pos| Mat4::from_translation(vec3(pos.x, pos.y, pos.z)))
                        .collect(),
                    ..Default::default()
                },
                &sphere,
            ),
            wireframe_material.clone(),
        ),
        edges: Gm::new(
            InstancedMesh::new(
                context,
                &Instances {
                    transformations: mesh
                        .edges()
                        .map(|e| {
                            let h = e.halfedge(false);
                            let mut ev = mesh.calc_halfedge_vector(h, &points);
                            let length = ev.magnitude();
                            ev /= length;
                            let ev = vec3(ev.x, ev.y, ev.z);
                            let start = points[h.tail(&mesh)];
                            let start = vec3(start.x, start.y, start.z);
                            Mat4::from_translation(start)
                                * Into::<Mat4>::into(Quat::from_arc(vec3(1.0, 0., 0.0), ev, None))
                                * Mat4::from_nonuniform_scale(length, 1., 1.)
                        })
                        .collect(),
                    ..Default::default()
                },
                &cylinder,
            ),
            wireframe_material,
        ),
    })
}

pub fn base_mesh_view(mesh: PolygonMesh, context: &Context) -> Option<MeshView> {
    let face_material = PhysicalMaterial::new_transparent(
        context,
        &CpuMaterial {
            albedo: Srgba::new(200, 200, 200, 25),
            ..Default::default()
        },
    );
    let mut wireframe_material = PhysicalMaterial::new_opaque(
        context,
        &CpuMaterial {
            albedo: Srgba::new_opaque(220, 50, 50),
            ..Default::default()
        },
    );
    wireframe_material.render_states.cull = Cull::Back;
    mesh_view(
        mesh,
        context,
        0.0025,
        0.0025,
        face_material,
        wireframe_material,
    )
}

pub fn surf_mesh_view(mesh: PolygonMesh, context: &Context) -> Option<MeshView> {
    let face_material = PhysicalMaterial::new_opaque(
        context,
        &CpuMaterial {
            albedo: Srgba::new_opaque(100, 200, 100),
            ..Default::default()
        },
    );
    let mut wireframe_material = PhysicalMaterial::new_opaque(
        context,
        &CpuMaterial {
            albedo: Srgba::new_opaque(50, 100, 50),
            ..Default::default()
        },
    );
    wireframe_material.render_states.cull = Cull::Back;
    mesh_view(
        mesh,
        context,
        0.0025,
        0.0025,
        face_material,
        wireframe_material,
    )
}

pub fn volume_mesh_view(mesh: PolygonMesh, context: &Context) -> Option<MeshView> {
    let face_material = PhysicalMaterial::new_opaque(
        context,
        &CpuMaterial {
            albedo: Srgba::new_opaque(100, 100, 200),
            ..Default::default()
        },
    );
    let mut wireframe_material = PhysicalMaterial::new_opaque(
        context,
        &CpuMaterial {
            albedo: Srgba::new_opaque(50, 50, 100),
            ..Default::default()
        },
    );
    wireframe_material.render_states.cull = Cull::Back;
    mesh_view(
        mesh,
        context,
        0.0025,
        0.0025,
        face_material,
        wireframe_material,
    )
}
