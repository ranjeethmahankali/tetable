use crate::tables::{self, SurfaceTable, VolumeTable};
use std::fmt::Write;
use std::path::Path;

pub fn export_tables(
    surf_table: &SurfaceTable,
    volume_table: &VolumeTable,
    path: &Path,
) -> Result<(), String> {
    std::fs::write(
        path,
        match path.extension().map(|e| e.to_str()).flatten() {
            Some("cpp" | "c" | "h" | "hpp") => cpp_tables(surf_table, volume_table),
            _ => {
                return Err("Unrecognized language / file extension".to_string());
            }
        }
        .map_err(|e| format!("{e}"))?,
    )
    .map_err(|e| format!("{e}"))
}

fn cpp_tables(
    surf_table: &SurfaceTable,
    volume_table: &VolumeTable,
) -> Result<String, std::fmt::Error> {
    let mut out = String::new();
    writeln!(out, "namespace tetable {{\n")?; // begin namespace.
    {
        // Surface - indices.
        let indices = surf_table.indices();
        writeln!(
            out,
            "static constexpr std::array<uint8_t, {}> SURF_INDICES {{{{",
            indices.len()
        )?;
        for chunk in indices.chunks(24) {
            write!(out, "\t")?;
            for i in chunk {
                write!(out, "{}, ", i)?;
            }
            writeln!(out, "")?;
        }
        writeln!(out, "}}}};\n")?; // end array.
        // Ranges.
        let ranges = surf_table.ranges();
        writeln!(
            out,
            "\nstatic constexpr std::array<std::pair<size_t, size_t>, {}> SURF_LOOKUP {{{{",
            ranges.len()
        )?;
        for chunk in ranges.chunks(8) {
            write!(out, "\t")?;
            for (start, len) in chunk {
                write!(out, "{{{start}, {len}}}, ")?;
            }
            writeln!(out, "")?;
        }
        writeln!(out, "}}}};\n")?; // end array.
    }
    {
        // Volume - indices.
        let indices = volume_table.indices();
        writeln!(
            out,
            "static constexpr std::array<uint8_t, {}> VOLUME_INDICES {{{{",
            indices.len()
        )?;
        for chunk in indices.chunks(24) {
            write!(out, "\t")?;
            for i in chunk {
                write!(out, "{}, ", i)?;
            }
            writeln!(out, "")?;
        }
        writeln!(out, "}}}};\n")?; // end array.
        // ranges.
        let ranges = volume_table.ranges();
        writeln!(
            out,
            "\nstatic constexpr std::array<std::array<std::pair<size_t, size_t>, {}>, {}> VOLUME_LOOKUP {{{{",
            tables::NUM_IDX_PERM,
            tables::NUM_MASK_COMB
        )?;
        for ranges in ranges {
            writeln!(out, "\t{{{{")?; // open inner array.
            for chunk in ranges.chunks(8) {
                write!(out, "\t\t")?;
                for (start, len) in chunk {
                    write!(out, "{{{start}, {len}}}, ")?;
                }
                writeln!(out, "")?;
            }
            writeln!(out, "\t}}}},")?; // close inner array.
        }
        writeln!(out, "}}}};\n")?; // end array.
    }
    writeln!(out, "}}")?; // end namespace.
    Ok(out)
}
