use clap::Parser;

#[derive(Parser)]
pub struct Cli {
    #[arg(long, default_value_t = false)]
    pub DEBUG: bool,

    #[arg(long, default_value_t = true)]
    pub do_train: bool,

    #[arg(long, default_value = "monkey-128-no-shading")]
    pub img_dir: String,

    #[arg(long, default_value = "logs")]
    pub log_dir: String,

    #[arg(long, default_value = "checkpoints")]
    pub save_dir: String,
    //checkpoint-1753406758-19000.ot
    // checkpoint-1753394531-47000.ot
    // checkpoint-1753331736-26000.ot
    // checkpoint-1753303468-8484.ot")
    // checkpoint-1718944888-11413.ot
    // checkpoint-1718941373-6161.ot
    #[arg(long, default_value = "")]
    pub load_path: String,

    #[arg(long, default_value_t = 50000)]
    pub num_iter: usize,

    #[arg(long, default_value_t = 360)]
    pub eval_steps: usize,

    #[arg(long, default_value_t = 1000)]
    pub save_steps: usize,

    #[arg(long, default_value_t = 100)]
    pub refresh_epochs: usize,
}
