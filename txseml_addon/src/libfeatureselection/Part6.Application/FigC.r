color_list_ <- list(
  T1 = "#903d32",
  T2 = "#4e8e47",
  T6 = "#3b328b"
)
for (prottype in c("T1", "T2", "T6")) {
  predict_result_text_count_df <- read.delim(
    file = sprintf("out/libfeatureselection/Part6.Application/out/data/FigC/%s.csv", prottype), # nolint: line_length_linter.
    sep = ",", header = TRUE, check.names = FALSE, row.names = 1
  )

  legend_breaks <- c(0., 1.)
  legend_label_breaks <- c(
    sprintf("non-%sSE", prottype), sprintf("%sSE", prottype)
  )
  color_list <- c("#bebebe", color_list_[[prottype]])

  p <- pheatmap::pheatmap(
    predict_result_text_count_df,
    cluster_rows = FALSE, cluster_cols = FALSE,
    show_colnames = FALSE,
    treeheight_col = 0,
    legend_breaks = legend_breaks,
    color = color_list,
    legend = FALSE,
    gaps_row = FALSE,
    gaps_col = FALSE,
  )

  ggplot2::ggsave(
    sprintf(
      "out/libfeatureselection/Part6.Application/out/Fig/FIg8C-%s.pdf",
      prottype
    ),
    plot = p,
    width = 58, height = 15,
    units = "cm",
  )
}
