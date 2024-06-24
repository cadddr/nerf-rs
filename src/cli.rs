use clap::Parser;

#[derive(Parser)]
pub struct Cli {
    #[arg(long, default_value_t = false)]
    pub DEBUG: bool,

    #[arg(long, default_value = "spheres-128-no-shading")]
    pub img_dir: String,

    #[arg(long, default_value = "logs")]
    pub log_dir: String,

    #[arg(long, default_value = "checkpoints")]
    pub save_dir: String,

    #[arg(long, default_value = "")] //checkpoints/checkpoint-1718836047-505.ot")]
    pub load_path: String,

    #[arg(long, default_value_t = 50000)]
    pub num_iter: usize,

    #[arg(long, default_value_t = 101)]
    pub eval_steps: usize,

    #[arg(long, default_value_t = 100)]
    pub refresh_epochs: usize,
}
