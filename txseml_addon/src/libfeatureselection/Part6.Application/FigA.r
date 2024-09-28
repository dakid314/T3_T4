predict_result_text_count_df <- read.delim(
  file = "out/libfeatureselection/Part6.Application/out/data/predict_result_text_count_df.csv", # nolint: line_length_linter.
  sep = ",", header = TRUE, check.names = FALSE,
)

color_list <- list(
  T1 = "#903d32",
  T2 = "#4e8e47",
  T6 = "#3b328b"
)

colnames(predict_result_text_count_df) <- c("ProtType", "Type", "Count")

plot_list <- list()
for (typeprot in unique(predict_result_text_count_df$ProtType)) {
  subset_df <- subset(predict_result_text_count_df, ProtType == typeprot)
  p <- ggplot2::ggplot(
    subset_df,
    ggplot2::aes(
      fill = Type, y = Count, x = ProtType,
    )
  ) +
    ggplot2::geom_text(ggplot2::aes(label = Count),
      position = ggplot2::position_fill(vjust = 0.5),
      size = 5
    ) +
    ggplot2::geom_bar(position = "stack", stat = "identity") +
    ggplot2::facet_wrap(~ProtType) +
    ggplot2::scale_fill_manual(values = c("#bebebe", color_list[[typeprot]])) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      axis.title.x = ggplot2::element_blank(),
      axis.ticks.x = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold", size = 14),
    ) +
    ggplot2::theme(
      legend.position = "none"
    )

  plot_list[[typeprot]] <- p
  ggplot2::ggsave(
    "out/libfeatureselection/Part6.Application/out/Fig/FIg8A.pdf",
    plot = cowplot::plot_grid(plotlist = plot_list, nrow = 1),
    width = 14, height = 15,
    units = "cm",
  )
}