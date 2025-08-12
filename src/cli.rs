use std::collections::HashMap;

use clap::{CommandFactory, Parser};

#[derive(Parser)]
pub struct Cli {
    #[arg(long, default_value_t = false)]
    pub debug: bool,

    #[arg(long, default_value_t = true)]
    pub do_train: bool,

    #[arg(long, default_value_t = true)]
    pub eval_on_train: bool,

    #[arg(long, default_value_t = false)]
    pub log_densities_only: bool,

    #[arg(long, default_value = "monkey-128-no-shading")]
    pub img_dir: String,

    #[arg(long, default_value_t = 0)]
    pub view_start_h: usize,

    #[arg(long, default_value_t = 360)]
    pub view_end_h: usize,

    #[arg(long, default_value_t = 10)]
    pub view_step_h: usize,

    #[arg(long, default_value = "logs")]
    pub log_dir: String,

    #[arg(long, default_value = "checkpoints")]
    pub save_dir: String,

    // checkpoint-1753511632-49049.ot
    // checkpoint-1753425343-50000.ot
    // checkpoint-1753411427-19000.ot
    // checkpoint-1753406758-19000.ot
    // checkpoint-1753394531-47000.ot
    // checkpoint-1753331736-26000.ot
    // checkpoint-1753303468-8484.ot")
    // checkpoint-1718944888-11413.ot
    // checkpoint-1718941373-6161.ot
    #[arg(long, default_value = "checkpoint-1753829839-5001.ot")]
    pub load_path: String,

    #[arg(long, default_value_t = 50000)]
    pub num_iter: usize,

    #[arg(long, default_value_t = 101)]
    pub eval_steps: usize,

    #[arg(long, default_value_t = 101)]
    pub logging_steps: usize,

    #[arg(long, default_value_t = 5001)]
    pub save_steps: usize,

    #[arg(long, default_value_t = 1)]
    pub accumulation_steps: usize,

    #[arg(long, default_value_t = 5e-4)]
    pub learning_rate: f64,
}

pub fn get_scalars_as_map() -> HashMap<String, f32> {
    let matches = Cli::command().get_matches();
    let mut args_map: HashMap<String, f32> = HashMap::new();

    for arg_id in matches.ids() {
        let key = arg_id.as_str();
        if let Ok(Some(value)) = matches.try_get_one::<usize>(key) {
            args_map.insert(key.to_string(), *value as f32);
        }
    }
    return args_map;
}
