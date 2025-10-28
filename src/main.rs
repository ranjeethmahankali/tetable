mod mesh;
mod tables;
mod export;

use std::time::Instant;
use mesh::{MeshView, PolygonMesh, base_mesh_view, surf_mesh_view, volume_mesh_view};
use tables::{SurfaceTable, VolumeTable};
use three_d::{
    AmbientLight, Camera, ClearState, Context, DirectionalLight, FrameOutput, InnerSpace, Object,
    OrbitControl, Srgba, Viewport, Window, WindowSettings, degrees, vec3,
};
use export::export_tables;

struct VertState {
    state: String,
    index: String,
}

#[derive(Default)]
enum State {
    #[default]
    None,
    Valid {
        mask: u8,
        rank: u8,
    },
    Changed,
    Error(String),
}

struct App {
    vstates: [VertState; 4],
    state: State,
    surf_table: SurfaceTable,
    volume_table: VolumeTable,
    base_tet: MeshView,
    surf_mesh: Option<MeshView>,
    surf_indices: Vec<u8>,
    vol_mesh: Option<MeshView>,
    vol_indices: Vec<u8>,
}

impl App {
    fn new(context: &Context) -> Self {
        let mut out = Self {
            vstates: [
                VertState {
                    state: "0".to_string(),
                    index: "0".to_string(),
                },
                VertState {
                    state: "0".to_string(),
                    index: "1".to_string(),
                },
                VertState {
                    state: "2".to_string(),
                    index: "2".to_string(),
                },
                VertState {
                    state: "2".to_string(),
                    index: "3".to_string(),
                },
            ],
            state: Default::default(),
            surf_table: generate_surf_table(),
            volume_table: generate_volume_table(),
            base_tet: base_mesh_view(
                PolygonMesh::tetrahedron(1.0).expect("Cannot create tet mesh"),
                &context,
            )
            .expect("Cannot create the base mesh view"),
            surf_mesh: None,
            surf_indices: Vec::new(),
            vol_mesh: None,
            vol_indices: Vec::new(),
        };
        out.state = State::Changed;
        out.update(context);
        out
    }

    fn objects(&self) -> impl Iterator<Item = &dyn Object> {
        self.base_tet
            .as_iter()
            .chain(self.surf_mesh.iter().flat_map(|m| m.as_iter()))
            .chain(self.vol_mesh.iter().flat_map(|m| m.as_iter()))
    }

    fn update(&mut self, context: &Context) -> bool {
        if let State::Changed = self.state {
            let mask = match self
                .vstates
                .iter()
                .enumerate()
                .try_fold(0u8, |acc, (i, vs)| {
                    let val = vs
                        .state
                        .parse::<u8>()
                        .map_err(|e| format!("ERROR: Unable to parse vertex state: {}", e))?;
                    if val > 2 {
                        return Err(format!(
                            "ERROR: The vertex state can be either 0, 1, or 2. {} is not valid",
                            val
                        ));
                    }
                    Ok(acc + 3u8.pow(i as u32) * val)
                }) {
                Ok(mask) => mask,
                Err(msg) => {
                    self.state = State::Error(msg);
                    return true;
                }
            };
            let rank = {
                let mut indices = [0usize; 4];
                for (src, dst) in self.vstates.iter().zip(indices.iter_mut()) {
                    *dst = match src.index.parse::<usize>() {
                        Ok(val) => val,
                        Err(e) => {
                            self.state = State::Error(format!(
                                "ERROR: Unable to parse the global vertex index: {}",
                                e
                            ));
                            return true;
                        }
                    }
                }
                let out = match tables::lehmer_rank(indices) {
                    Ok(rank) => rank,
                    Err(msg) => {
                        self.state = State::Error(msg.to_string());
                        return true;
                    }
                };
                indices.sort();
                if indices.windows(2).any(|w| w[0] >= w[1]) {
                    self.state =
                        State::Error("ERROR: The global vertex indices must be unique".to_string());
                    return true;
                }
                out
            };
            // Update surface mesh.
            let (mesh, indices) = self.surf_table.lookup(mask);
            self.surf_mesh = surf_mesh_view(mesh, &context);
            self.surf_indices = indices;
            // Update volume mesh.
            let (mesh, indices) = self.volume_table.lookup(mask, rank, 0.8);
            self.vol_mesh = volume_mesh_view(mesh, &context);
            self.vol_indices = indices;
            // Update state.
            self.state = State::Valid { mask, rank };
            return true;
        }
        return false;
    }
}

fn generate_surf_table() -> SurfaceTable {
    let before = Instant::now();
    let table = SurfaceTable::generate();
    let duration = Instant::now() - before;
    println!("Surface table generation took {}µs", duration.as_micros());
    table
}

fn generate_volume_table() -> VolumeTable {
    let before = Instant::now();
    let table = VolumeTable::generate();
    let duration = Instant::now() - before;
    println!("Volume table generation took {}µs", duration.as_micros());
    table
}

fn main() {
    // Window and context.
    let window = Window::new(WindowSettings {
        title: "Example: Rendering a Triangle Mesh".to_string(),
        min_size: (512, 256),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();
    let mut app = App::new(&context);
    // Setup the camera and the controls and lights.
    let target = vec3(0., 0., 0.);
    let scene_radius: f32 = 6.0;
    let mut camera = Camera::new_perspective(
        window.viewport(),
        target + scene_radius * vec3(0.6, 0.3, 1.0).normalize(),
        target,
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(camera.target(), 1.0, 100.0);
    let ambient = AmbientLight::new(&context, 0.7, Srgba::WHITE);
    let directional0 = DirectionalLight::new(&context, 2.0, Srgba::WHITE, vec3(-1.0, -1.0, -1.0));
    let directional1 = DirectionalLight::new(&context, 2.0, Srgba::WHITE, vec3(1.0, 1.0, 1.0));
    // Render loop.
    let mut gui = three_d::GUI::new(&context);
    window.render_loop(move |mut frame_input| {
        let mut panel_width = 0.0;
        let mut gui_wants_pointer = false;
        let mut app_changed = false;
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                use three_d::egui::*;
                // Calculate panel widths as a ratio of the viewport width
                let viewport_width =
                    frame_input.viewport.width as f32 / frame_input.device_pixel_ratio;
                const RATIO: f32 = 0.15;
                panel_width = viewport_width * RATIO;
                let left_response = SidePanel::left("inputs_panel")
                    .exact_width(panel_width)
                    .resizable(false)
                    .show(gui_context, |ui| {
                        ui.heading("Vertex Info");
                        ui.add_space(10.0);
                        Grid::new("input_table")
                            .striped(true)
                            .num_columns(3)
                            .show(ui, |ui| {
                                // Header row
                                ui.label("");
                                ui.label("state");
                                ui.label("index");
                                ui.end_row();
                                // Data rows
                                for i in 0..4 {
                                    ui.label(i.to_string());
                                    let response_left = ui
                                        .add(TextEdit::singleline(&mut app.vstates[i].state));
                                    let response_right = ui
                                        .add(TextEdit::singleline(&mut app.vstates[i].index));
                                    ui.end_row();
                                    if response_left.changed() || response_right.changed() {
                                        app.state = State::Changed;
                                    }
                                    // Trigger redraw when focus changes (e.g., Tab pressed)
                                    if response_left.gained_focus() || response_right.gained_focus() {
                                        app_changed = true;
                                    }
                                }
                            });
                        // Help info.
                        ui.separator();
                        ui.add_space(10.);
                        ui.heading("Help");
                        ui.label(RichText::new(
                            r#"
Please populate the vertex information in the table above.

For each vertex, the state can be either 0, 1, or 2. The values, in that order, indicate whether the vertex is inside the surface, exactly on the surface, or outside the surface.

The index of each vertex must be the global index of that vertex in the overall background tet-grid. This is used to compute the Lehmer Rank of the tetrahedron. This ensures that surface triangulation is consistent with the neighboring tetrahdra.
"#).size(14.),
                        );
                        ui.add_space(10.);
                        app_changed |= app.update(&context);
                        ui.heading("Outputs");
                        ui.add_space(10.0);
                        match &app.state {
                            State::None => ui.label(RichText::new("NO OUTPUT").size(14.)),
                            State::Valid { mask, rank } => {
                                ui.label(
                                        RichText::new(format!("Mask: {}", mask)).size(14.),
                                );
                                ui.add_space(10.);
                                ui.label(
                                        RichText::new(format!("Lehmer Rank: {}", rank)).size(14.),
                                );
                                ui.add_space(10.);
                                ui.label(RichText::new(format!("Surface indices:\n{:?}", app.surf_indices)).size(14.));
                                ui.label(RichText::new(format!("The count is {}, that means the surface has {} triangles.",
                                                               app.surf_indices.len(), app.surf_indices.len() / 3))
                                         .size(14.));
                                ui.add_space(10.);
                                ui.label(RichText::new(format!("Volume indices:\n{:?}", app.vol_indices)).size(14.));
                                let n_vol = app.vol_indices.len();
                                if n_vol > 0 {
                                    ui.label(RichText::new(format!("The count is {}, that means the surface has {} tetrahedra.",
                                                                   app.vol_indices.len(), app.vol_indices.len() / 4))
                                             .size(14.))
                                        
                                } else {
                                    ui.label(RichText::new(format!("The volume indices are empty, that means this tetrahedron should not be split into smaller tetrahedra")))
                                }
                            },
                            State::Changed => panic!("The app state was not updated"),
                            State::Error(msg) => ui.monospace(msg), 
                        };
                        // Save button.
                        ui.separator();
                        ui.add_space(10.0);
                        if ui.button("Export Tables").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("C++", &["cpp", "hpp"])
                                .save_file()
                            {
                                match export_tables(&app.surf_table, &app.volume_table, &path) {
                                    Ok(()) => ui.label(format!("Tables were written to {}", path.display())),
                                    Err(e) => ui.monospace(format!("ERROR: {}", e)),
                                };
                            }
                        }
                    });
                panel_width = left_response.response.rect.width();
                gui_wants_pointer = gui_context.wants_pointer_input();
            },
        );
        let mut redraw = frame_input.first_frame || app_changed;
        redraw |= camera.set_viewport(Viewport {
            x: (panel_width * frame_input.device_pixel_ratio) as i32,
            y: 0,
            width: frame_input.viewport.width
                - ((panel_width * frame_input.device_pixel_ratio)
                    as u32),
            height: frame_input.viewport.height,
        });
        if gui_wants_pointer  {
            redraw = true;
        } else {
            redraw |= control.handle_events(&mut camera, &mut frame_input.events);
        }
        if redraw {
            frame_input
                .screen()
                .clear(ClearState::color_and_depth(0.1, 0.1, 0.1, 1.0, 1.0))
                .write(|| {
                    for obj in app.objects() {
                        obj.render(&camera, &[&ambient, &directional0, &directional1]);
                    }
                    gui.render()
                })
                .expect("Cannot render the scene");
        }
        FrameOutput {
            swap_buffers: redraw,
            ..Default::default()
        }
    });
}
