use crate::mesh::PolygonMesh;
use alum::{EditableTopology, FH, HH, Handle, HasIterators, HasTopology, Status, VH, VPropBuf};
use three_d::Vec3;

pub fn lehmer_rank(indices: [usize; 4]) -> Result<u8, &'static str> {
    const FACTORIAL: [u8; 4] = [6, 2, 1, 0];
    match indices
        .iter()
        .zip(FACTORIAL.iter())
        .enumerate()
        .map(|(i, (first, factorial))| {
            (indices
                .iter()
                .skip(i + 1)
                .filter(|second| first > second)
                .count() as u8)
                * factorial
        })
        .reduce(|a, b| a + b)
    {
        Some(rank) => Ok(rank),
        None => Err("ERROR: Unable to compute the Lehmer Rank"),
    }
}

fn index_permutations() -> impl Iterator<Item = [usize; 4]> {
    (0usize..4).flat_map(|a| {
        (0usize..4).filter(move |b| *b != a).flat_map(move |b| {
            (0usize..4)
                .filter(move |c| *c != b && *c != a)
                .flat_map(move |c| {
                    (0usize..4)
                        .filter(move |d| *d != a && *d != b && *d != c)
                        .map(move |d| [a, b, c, d])
                })
        })
    })
}

fn base_tet() -> PolygonMesh {
    let mut mesh = PolygonMesh::tetrahedron(1.0).expect("Cannot create a tetrahedron");
    for ei in 0u32..6 {
        let (v0, v1) = edge_vertices(ei);
        let p0 = mesh.point(v0).expect("cannot retrieve point");
        let p1 = mesh.point(v1).expect("cannot retrieve point");
        mesh.add_vertex((p0 + p1) * 0.5)
            .expect("Unable to add edge vertices to mesh");
    }
    mesh
}

fn base_tet_no_faces() -> PolygonMesh {
    let mut mesh = base_tet();
    for f in mesh.faces() {
        mesh.delete_face(f, false).expect("Cannot delete face");
    }
    mesh
}

#[derive(Debug, Clone, Copy)]
enum CrossingType {
    Inside,
    InsideVertexIncident,
    InsideEdgeIncident,
    InsideFaceIncident,
    DegenerateIncident, // All tet vertices are on the surface.
    Intersecting,
    OutsideFaceIncident,
    OutsideEdgeIncident,
    OutsideVertexIncident,
    Outside,
}
use CrossingType::*;

fn unpack_mask(mut mask: u8) -> ([u8; 4], CrossingType) {
    let (unpacked, inside, on_surf, outside) = {
        let mut out = [0u8; 4];
        let mut inside = 0usize;
        let mut on_surf = 0usize;
        let mut outside = 0usize;
        for dst in out.iter_mut() {
            *dst = mask % 3;
            mask /= 3;
            match *dst {
                0 => inside += 1,
                1 => on_surf += 1,
                2 => outside += 1,
                _ => panic!("Invalid unpacked mask index"),
            }
        }
        (out, inside, on_surf, outside)
    };
    assert_eq!(inside + on_surf + outside, 4);
    let crossing = match (inside, on_surf, outside) {
        (4, 0, 0) => Inside,
        (3, 1, 0) => InsideVertexIncident,
        (2, 2, 0) => InsideEdgeIncident,
        (1, 3, 0) => InsideFaceIncident,
        (0, 4, 0) => DegenerateIncident,
        (0, 3, 1) => OutsideFaceIncident,
        (0, 2, 2) => OutsideEdgeIncident,
        (0, 1, 3) => OutsideVertexIncident,
        (0, 0, 4) => Outside,
        _ => Intersecting,
    };
    (unpacked, crossing)
}

fn split_edges_by_mask(
    cross_type: CrossingType,
    unpacked_mask: [u8; 4],
    mesh: &mut PolygonMesh,
    vstatus: &mut VPropBuf<Status>,
) {
    for v in mesh.vertices().take(4) {
        match unpacked_mask[v.index() as usize] {
            0 => {
                // Inside
                vstatus[v].set_tagged(false);
                vstatus[v].set_feature(false);
            }
            1 => {
                // On surface.
                vstatus[v].set_tagged(false);
                vstatus[v].set_feature(true);
            }
            2 => {
                // Outside.
                vstatus[v].set_tagged(true);
                vstatus[v].set_feature(false);
            }
            _ => panic!("Invalid mask"),
        }
    }
    if let Intersecting = cross_type {
        for e in mesh.edges() {
            let (from, to) = e.vertices(mesh);
            if vstatus[from].tagged() != vstatus[to].tagged()
                && !vstatus[from].feature()
                && !vstatus[to].feature()
            {
                let vedge: VH = (edge_from_verts(from, to) + 4).into();
                vstatus[vedge].set_feature(true);
                assert!(vedge.halfedge(mesh).is_none(), "Vertex must be isolated");
                mesh.split_edge(e.halfedge(true), vedge, true)
                    .expect("Cannot split a crossing edge.");
            }
        }
        for f in mesh.faces() {
            if f.valence(mesh) == 3 {
                continue;
            }
            let (hprev, hnext) = {
                let mut hiter = mesh
                    .fh_ccw_iter(f)
                    .filter(|h| vstatus[h.head(mesh)].feature());
                let h0 = hiter
                    .next()
                    .expect("Polygon face doesn't have a feature edge for splitting");
                let h1 = hiter
                    .next()
                    .expect("Polygon face doesn't have a second feature edge for splitting");
                if hiter.next().is_some() {
                    panic!("Polygon face has too many feature vertices");
                }
                (h0, h1.next(mesh))
            };
            let edge = mesh
                .insert_edge(hprev, hnext, None)
                .expect("Cannot insert edge to split the polygon face");
            mesh.edge_status_mut(edge)
                .expect("Cannot borrow edge status")
                .set_feature(true);
        }
    }
}

fn edge_from_verts(mut v0: VH, mut v1: VH) -> u32 {
    assert!(
        v0.index() < 4 && v1.index() < 4,
        "Vertex indices must be in range [0,4)"
    );
    assert!(v0 != v1, "Vertices must be different");
    if v0 > v1 {
        std::mem::swap(&mut v0, &mut v1);
    }
    // For vertices (v0, v1) where v0 < v1:
    // Number of edges from vertices < v0 plus offset within v0's edges
    v0.index() * (7 - v0.index()) / 2 + (v1.index() - v0.index() - 1)
}

fn edge_vertices(edge: u32) -> (VH, VH) {
    assert!(edge < 6, "Invalid edge index for tetrahedron");
    // Find v0 by checking cumulative edge counts
    // Vertex 0: edges 0,1,2 (3 edges)
    // Vertex 1: edges 3,4 (2 edges)
    // Vertex 2: edge 5 (1 edge)
    let v0 = if edge < 3 {
        0
    } else if edge < 5 {
        1
    } else {
        2
    };
    // Compute offset within v0's edges
    let offset = edge - (v0 * (7 - v0) / 2);
    let v1 = v0 + offset + 1;
    assert!(v0 < 4 && v1 < 4);
    (v0.into(), v1.into())
}

pub const NUM_IDX_PERM: usize = 24;
pub const NUM_MASK_COMB: usize = 81;

pub struct SurfaceTable {
    ranges: [(usize, usize); NUM_MASK_COMB],
    indices: Vec<u8>,
}

pub struct VolumeTable {
    ranges: [[(usize, usize); NUM_IDX_PERM]; NUM_MASK_COMB],
    indices: Vec<u8>,
}

impl SurfaceTable {
    pub fn generate() -> Self {
        let mut ranges = [(0usize, 0usize); NUM_MASK_COMB];
        let mut indices = Vec::<u8>::new();
        let mut vcache = Vec::new();
        for (index, range) in ranges.iter_mut().enumerate() {
            let mask = index as u8;
            let (unpacked, cross_type) = unpack_mask(mask);
            let mut mesh = base_tet();
            let h = match cross_type {
                Inside
                | InsideVertexIncident
                | InsideEdgeIncident
                | DegenerateIncident
                | OutsideFaceIncident
                | OutsideEdgeIncident
                | OutsideVertexIncident
                | Outside => continue,
                InsideFaceIncident => {
                    let mut vstatus = mesh.vertex_status_prop();
                    let mut vstatus = vstatus
                        .try_borrow_mut()
                        .expect("Cannot borrow vertex status");
                    split_edges_by_mask(cross_type, unpacked, &mut mesh, &mut vstatus);
                    mesh.halfedges().find(|h| {
                        !mesh
                            .halfedge_status(*h)
                            .expect("Cannot get halfedge status")
                            .deleted()
                            && vstatus[h.head(&mesh)].feature()
                            && vstatus[h.tail(&mesh)].feature()
                            && vstatus[h.next(&mesh).head(&mesh)].feature()
                    }).expect("Cannot find halfedge spanning two feature vertices. That was expected.")
                }
                Intersecting => {
                    {
                        let mut vstatus = mesh.vertex_status_prop();
                        let mut vstatus = vstatus
                            .try_borrow_mut()
                            .expect("Cannot borrow vertex status");
                        split_edges_by_mask(cross_type, unpacked, &mut mesh, &mut vstatus);
                        vcache.clear();
                        vcache.extend(mesh.vertices().filter(|v| vstatus[*v].tagged()));
                    }
                    for v in vcache.drain(..) {
                        mesh.delete_vertex(false, v)
                            .expect("Unable to delete vertex");
                    }
                    mesh.halfedges().find(|h| {
                        h.is_boundary(&mesh)
                            && !mesh
                            .halfedge_status(*h)
                            .expect("Cannot get halfedge status")
                            .deleted()
                    }).expect("Cannot find boundary edge for a tet that is expected to have been sliced open.")
                }
            };
            let start = indices.len();
            indices.extend(
                mesh.triangulated_loop_vertices(h)
                    .flatten()
                    .map(|v| v.index() as u8),
            );
            let len = indices.len() - start;
            assert!(len % 3 == 0, "Number of indices must be a multiple of 3");
            *range = (start, len);
        }
        SurfaceTable { ranges, indices }
    }

    pub fn indices(&self) -> &[u8] {
        &self.indices
    }

    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }

    pub fn lookup(&self, mask: u8) -> (PolygonMesh, Vec<u8>) {
        let indices = {
            let (start, len) = self.ranges[mask as usize];
            &self.indices[start..(start + len)]
        };
        assert!(indices.len() % 3 == 0, "Must be a multiple of 3");
        let mut mesh = base_tet_no_faces();
        for chunk in indices.chunks_exact(3) {
            let a = chunk[0];
            let b = chunk[1];
            let c = chunk[2];
            mesh.add_tri_face((a as u32).into(), (b as u32).into(), (c as u32).into())
                .expect("Cannot add face");
        }
        mesh.garbage_collection()
            .expect("Failed to garbage collect");
        (mesh, indices.to_vec())
    }
}

fn triangulate_polyhedron(
    mesh: &mut PolygonMesh,
    perm: [usize; 4],
    fcache: &mut Vec<FH>,
    hcache: &mut Vec<HH>,
) {
    // Must be either a tet or a prism.
    {
        let fstatus = mesh.face_status_prop();
        let fstatus = fstatus.try_borrow().expect("Cannot borrow face status");
        match mesh.faces().filter(|f| !fstatus[*f].deleted()).fold(
            (0usize, 0usize),
            |(tris, quads), f| match f.valence(mesh) {
                3 => (tris + 1, quads),
                4 => (tris, quads + 1),
                _ => panic!("Invalid face. Only expecting triangles and quads"),
            },
        ) {
            (4, 0) | (2, 3) | (4, 1) | (4, 2) => {} // Valid.
            (0, 0) => return,
            (tris, quads) => {
                eprintln!("Tri count: {tris}; Quad count: {quads}");
                panic!("The polyhedron must be either a tetrahedron or a prism. It is neither")
            }
        };
    }
    {
        fcache.clear();
        let fstatus = mesh.face_status_prop();
        let fstatus = fstatus
            .try_borrow()
            .expect("Unable to borrow face properties");
        fcache.extend(
            mesh.faces()
                .filter(|f| !fstatus[*f].deleted() && f.valence(mesh) > 3),
        );
    }
    for f in fcache.drain(..) {
        assert_eq!(f.valence(mesh), 4, "Expecting only quad faces");
        hcache.clear();
        hcache.extend(mesh.fh_ccw_iter(f).filter(|h| h.head(mesh).index() < 4));
        assert_eq!(
            hcache.len(),
            2,
            "Quad face should have exactly two original vertices"
        );
        hcache.sort_by_key(|h| perm[h.head(mesh).index() as usize]);
        let hprev = hcache
            .pop()
            .expect("Unable to get the preferred edge for triangulating the quad");
        let hnext = hprev.prev(mesh);
        mesh.insert_edge(hprev, hnext, None)
            .expect("Unable to triangulate quad");
    }
}

fn extract_one_tet(
    mesh: &mut PolygonMesh,
    v: VH,
    vcache: &mut Vec<VH>,
    fcache: &mut Vec<FH>,
) -> [u8; 4] {
    vcache.clear();
    fcache.clear();
    for h in mesh.voh_ccw_iter(v) {
        vcache.push(h.head(mesh));
        fcache.push(h.face(mesh).expect("The mesh must be closed"));
    }
    assert_eq!(vcache.len(), 3, "This must be a valence 3 vertex.");
    assert_eq!(fcache.len(), 3, "This must be a valence 3 vertex.");
    for f in fcache.drain(..) {
        mesh.delete_face(f, false).expect("Unable to delete face");
    }
    mesh.add_face(&vcache)
        .expect("Unable to add a new face during tetrahedralization");
    [
        vcache[0].index() as u8,
        vcache[2].index() as u8,
        vcache[1].index() as u8,
        v.index() as u8,
    ]
}

fn tetrahedralize_polyhedron(
    mut mesh: PolygonMesh,
    dst: &mut Vec<u8>,
    vcache: &mut Vec<VH>,
    fcache: &mut Vec<FH>,
) {
    let n_expected_tets = match {
        // Count triangles.
        let fstatus = mesh.face_status_prop();
        let fstatus = fstatus.try_borrow().expect("Cannot borrow face status");
        mesh.faces()
            .filter(|f| !fstatus[*f].deleted())
            .fold(0usize, |count, f| match f.valence(&mesh) {
                3 => count + 1,
                _ => panic!("Invalid face. Only expecting triangles"),
            })
    } {
        0 => return,
        4 => 1, // Tet
        6 => 2, // Pyramid
        8 => 3, // Prism
        _ => panic!(
            "The polyhedron must be either a tetrahedron or a prism, or a pyramid. It is neither"
        ),
    };
    let before = dst.len();
    while let Some(v) = mesh
        .vertices()
        .find(|v| !v.is_boundary(&mesh) && v.valence(&mesh) == 3)
    {
        let indices = extract_one_tet(&mut mesh, v, vcache, fcache);
        dst.extend_from_slice(&indices);
    }
    assert_eq!(
        (dst.len() - before) / 4,
        n_expected_tets,
        "Incorrect number of tets"
    );
}

impl VolumeTable {
    pub fn generate() -> Self {
        let mut ranges = [[(0usize, 0usize); NUM_IDX_PERM]; NUM_MASK_COMB];
        let mut indices = Vec::<u8>::new();
        let mut voutside = Vec::<VH>::new();
        let mut vinside = Vec::<VH>::new();
        let mut triplets = Vec::<[VH; 3]>::new();
        let mut fcache = Vec::new();
        let mut hcache = Vec::new();
        let mut vcache = Vec::new();
        for (mask_idx, comb_ranges) in ranges.iter_mut().enumerate() {
            let mask = mask_idx as u8;
            let (unpacked, cross_type) = unpack_mask(mask);
            match cross_type {
                Inside
                | InsideVertexIncident
                | InsideEdgeIncident
                | InsideFaceIncident
                | DegenerateIncident
                | OutsideFaceIncident
                | OutsideEdgeIncident
                | OutsideVertexIncident
                | Outside => continue,
                Intersecting => {
                    for iperm in index_permutations() {
                        let rank = lehmer_rank(iperm).expect("Unable to compute lehmer rank");
                        let range = &mut comb_ranges[rank as usize];
                        let mut mesh = base_tet();
                        {
                            let mut vstatus = mesh.vertex_status_prop();
                            let mut vstatus = vstatus
                                .try_borrow_mut()
                                .expect("Cannot borrow vertex status");
                            split_edges_by_mask(cross_type, unpacked, &mut mesh, &mut vstatus);
                            voutside.clear();
                            voutside.extend(mesh.vertices().filter(|v| vstatus[*v].tagged()));
                            vinside.clear();
                            vinside.extend(mesh.vertices().filter(|v| {
                                let vs = vstatus[*v];
                                !vs.tagged() && !vs.feature()
                            }));
                        }
                        let mut omesh = mesh.try_clone().expect("Unable to clone the mesh");
                        let mut imesh = mesh;
                        for v in voutside.drain(..) {
                            imesh
                                .delete_vertex(false, v)
                                .expect("Unable to delete vertex");
                        }
                        for v in vinside.drain(..) {
                            omesh
                                .delete_vertex(false, v)
                                .expect("Unable to delete vertex");
                        }
                        // Get the triangulation of the hole of inside mesh.
                        triplets.clear();
                        let h = imesh.halfedges().find(|h| {
                            h.is_boundary(&imesh)
                                && !imesh
                                    .halfedge_status(*h)
                                    .expect("Cannot get halfedge status")
                                    .deleted()
                        }).expect("Cannot find boundary edge for a tet that is expected to have been sliced open");
                        triplets.extend(imesh.triangulated_loop_vertices(h));
                        // Fill the holes.
                        for [a, b, c] in triplets.drain(..) {
                            imesh
                                .add_tri_face(a, b, c)
                                .expect("Unable to create sliced polyhedron");
                            omesh
                                .add_tri_face(a, c, b)
                                .expect("Unable to create sliced polyhedron");
                        }
                        let start = indices.len();
                        for mut m in [imesh, omesh] {
                            triangulate_polyhedron(&mut m, iperm, &mut fcache, &mut hcache);
                            tetrahedralize_polyhedron(m, &mut indices, &mut vcache, &mut fcache);
                        }
                        let len = indices.len() - start;
                        assert!(len % 4 == 0, "Number of indices must be a multiple of 4");
                        *range = (start, len);
                    }
                }
            }
        }
        VolumeTable { ranges, indices }
    }

    pub fn indices(&self) -> &[u8] {
        &self.indices
    }

    pub fn ranges(&self) -> &[[(usize, usize); NUM_IDX_PERM]] {
        &self.ranges
    }

    pub fn lookup(&self, mask: u8, rank: u8, scaling: f32) -> (PolygonMesh, Vec<u8>) {
        let indices = {
            let (start, len) = self.ranges[mask as usize][rank as usize];
            &self.indices[start..(start + len)]
        };
        assert!(indices.len() % 4 == 0, "Must be a multiple of 4");
        let points: Box<[Vec3]> = base_tet_no_faces()
            .points()
            .try_borrow()
            .expect("Cannot borrow points from base tet")
            .to_vec()
            .into_boxed_slice();
        let mut mesh = PolygonMesh::new();
        let mut ptcache: Vec<Vec3> = Vec::new();
        let mut verts: Vec<VH> = Vec::new();
        for chunk in indices.chunks_exact(4) {
            ptcache.clear();
            ptcache.extend(chunk.iter().map(|i| points[*i as usize]));
            let centroid = ptcache.iter().fold(Vec3::new(0., 0., 0.), |sum, v| sum + v)
                / (ptcache.len() as f32);
            for p in ptcache.iter_mut() {
                *p = scaling * *p + centroid * (1. - scaling);
            }
            verts.clear();
            verts.extend(
                ptcache
                    .iter()
                    .map(|p| mesh.add_vertex(*p).expect("Unable to add vertex to mesh")),
            );
            for [a, b, c] in [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]] {
                mesh.add_tri_face(verts[a], verts[b], verts[c])
                    .expect("Cannot add tet face to split-tet-mesh");
            }
        }
        (mesh, indices.to_vec())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use alum::{Handle, HasIterators, HasTopology};
    use three_d::{InnerSpace, Vec3};

    #[test]
    fn t_index_permutations() {
        let mut permutations: Vec<_> = index_permutations().collect();
        assert_eq!(permutations.len(), 24);
        permutations.sort();
        permutations.dedup();
        assert_eq!(permutations.len(), 24);
    }

    #[test]
    fn t_edge_indexing() {
        // Expected canonical edge ordering for a tetrahedron
        let expected = [(0u32, 1u32), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        // Test edge_vertices produces correct pairs
        for (e, &(v0, v1)) in expected.iter().enumerate() {
            assert_eq!(edge_vertices(e as u32), (v0.into(), v1.into()));
        }
        // Test edge_index produces correct indices
        for (e, &(v0, v1)) in expected.iter().enumerate() {
            assert_eq!(edge_from_verts(v0.into(), v1.into()), (e as u32).into());
            // Test order independence
            assert_eq!(edge_from_verts(v1.into(), v0.into()), (e as u32).into());
        }
        // Test round-trip conversion for all 6 edges
        for e in 0..6 {
            let (v0, v1) = edge_vertices(e);
            assert!(v0 < v1, "Vertices must be ordered");
            assert!(
                v0.index() < 4 && v1.index() < 4,
                "Vertices must be in range [0,4)"
            );
            assert_eq!(
                edge_from_verts(v0, v1),
                e,
                "Round-trip failed for edge {}",
                e
            );
        }
    }

    #[test]
    fn t_generate_surf_table() {
        let table = SurfaceTable::generate();
        let base_pts: Box<[Vec3]> = {
            let mesh = base_tet();
            let points = mesh.points();
            let points = points
                .try_borrow()
                .expect("Cannot borrow points of the base tet");
            points.iter().take(4).copied().collect()
        };
        // Check the orientaitons of the faces.
        let mut fvs: Vec<Vec3> = Vec::new();
        for mask in 0u8..81 {
            let (unpacked, _) = unpack_mask(mask);
            let (surfmesh, indices) = table.lookup(mask);
            let surf_pts = surfmesh.points();
            let surf_pts = surf_pts.try_borrow().expect("Cannot borrow points");
            assert_eq!(surfmesh.num_faces(), indices.len() / 3);
            assert_eq!(indices.len() % 3, 0);
            for (m, base_pt) in unpacked.iter().zip(base_pts.iter()) {
                for f in surfmesh.faces() {
                    fvs.clear();
                    fvs.extend(surfmesh.fv_ccw_iter(f).map(|v| surf_pts[v]));
                    assert_eq!(fvs.len(), 3);
                    let orientated_volume = (fvs[1] - fvs[0])
                        .cross(fvs[2] - fvs[1])
                        .dot(base_pt - fvs[0]);
                    match m {
                        0 => assert!(orientated_volume < 0.),
                        1 => assert!(orientated_volume.abs() <= f32::EPSILON),
                        2 => assert!(orientated_volume > 0.),
                        _ => panic!("Unexpected entry in unpacked mask"),
                    }
                }
            }
        }
    }

    #[test]
    fn t_generate_volume_table() {
        let table = VolumeTable::generate();
        let base_volume = base_tet()
            .try_calc_volume()
            .expect("Unable to compute the volume of the base tet mesh");
        for mask in 0u8..81 {
            let (_, cross_type) = unpack_mask(mask);
            for perm in index_permutations() {
                let rank = lehmer_rank(perm).expect("Unable to compute lehmer rank");
                let (vmesh, indices) = table.lookup(mask, rank, 1.0);
                assert_eq!(
                    vmesh.num_faces(),
                    indices.len(),
                    "4 faces per tet, and 4 vertice per tet, so they should be equal"
                );
                assert_eq!(indices.len() % 4, 0);
                if let Intersecting = cross_type {
                    assert!(
                        (base_volume
                            - vmesh
                                .try_calc_volume()
                                .expect("Unable to compute the volume of the split tet mesh"))
                        .abs()
                            <= f32::EPSILON
                    )
                };
            }
        }
    }
}
