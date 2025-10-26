mod mesh;

use mesh::{PolygonMesh, mesh_view, wireframe_view};
use three_d::{
    AmbientLight, Camera, ClearState, DirectionalLight, FrameOutput, InnerSpace, OrbitControl,
    Srgba, Viewport, Window, WindowSettings, degrees, vec3,
};

#[derive(Default)]
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

#[derive(Default)]
struct App {
    vstates: [VertState; 4],
    state: State,
}

impl App {
    fn update(&mut self) -> bool {
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
                let mut indices = [0u8; 4];
                for (src, dst) in self.vstates.iter().zip(indices.iter_mut()) {
                    *dst = match src.index.parse::<u8>() {
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
                const FACTORIAL: [u8; 4] = [6, 2, 1, 0];
                let out = match indices
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
                    Some(rank) => rank,
                    None => {
                        self.state =
                            State::Error("ERROR: Unable to compute the Lehmer Rank".to_string());
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
            self.state = State::Valid { mask, rank };
            return true;
        }
        return false;
    }
}

fn main() {
    // Window and context.
    let window = Window::new(WindowSettings {
        title: "Example: Rendering a Triangle Mesh".to_string(),
        min_size: (512, 256),
        ..Default::default()
    })
    .unwrap();
    let mut app = App::default();
    let context = window.gl();
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
    // Create the mesh.
    let mut mesh = PolygonMesh::tetrahedron(1.0).expect("Cannot create tet mesh");
    mesh.update_vertex_normals_accurate().unwrap();
    let view = mesh_view(&mesh, &context);
    let (vertices, edges) = wireframe_view(&mesh, &context, 0.01, 0.005);
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
                        ui.add_space(20.);
                        ui.heading("Help");
                        ui.label(RichText::new(
                            r#"
Please populate the vertex information in the table above.

For each vertex, the state can be either 0, 1, or 2. The values, in that order, indicate whether the vertex is inside the surface, exactly on the surface, or outside the surface.

The index of each vertex must be the global index of that vertex in the overall background tet-grid. This is used to compute the Lehmer Rank of the tetrahedron. This ensures that surface triangulation is consistent with the neighboring tetrahdra.
"#).size(14.),
                        );
                        ui.add_space(20.);
                        ui.heading("Outputs");
                        ui.add_space(10.0);
                        match &app.state {
                            State::None => ui.label(RichText::new("NO OUTPUT").size(14.)),
                            State::Valid { mask, rank } => {
                                ui.label(
                                        RichText::new(format!("Mask: {}", mask)).size(14.),
                                );
                                ui.label(
                                        RichText::new(format!("Lehmer Rank: {}", rank)).size(14.),
                                    )
                            },
                            State::Changed => panic!("The app state was not updated"),
                            State::Error(msg) => ui.monospace(msg),
                        }
                    });
                app_changed |= app.update();
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
                    for obj in view.into_iter().chain(&vertices).chain(&edges) {
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
