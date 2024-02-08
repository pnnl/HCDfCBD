#' Plot the variable importance based on a table of importance values and a table of raw feature values.
#' 
#' @param scores_df A dataframe with a column for the feature name in the first position, a column for the overall importance value in the second position, and a column for each sample with their importance values for each feature.
#' @param raw_data_df A dataframe with a column for the feature name (id_vars) and a column for each sample with their feature values.
#' @param num_biomols The number of biomolecules to plot.
#' @param id_vars The name of the column in raw_data_df that contains the feature names.
#' @param var_name What to name the column that contains the keys after calling \code{dplyr::pivot_longer} on `raw_data_df` and `scores_df`.
#' @param value_name What to name the column that contains the values after calling \code{dplyr::pivot_longer} on `raw_data_df`.
#' @param score_name What to name the column that contains the importance values after calling \code{dplyr::pivot_longer} on `scores_df`.
#' @param palette The name of the RColorBrewer color palette to use for the plot.
#' @param xlabel The label for the x-axis.
#' @param ylabel The label for the y-axis.
#' @param title The title for the plot.
#' @param draw_violin Whether to overlay a violin plot.
#' @param use_beeswarm Whether to use the \code{ggbeeswarm::geom_beeswarm} function instead of \code{ggplot2::geom_jitter}.
#' 
#' @return A ggplot object.
#' 
#' @importFrom ggplot2 aes
#' 
importance_plot <- function(
    scores_df,
    raw_data_df,
    num_biomols = 10,
    id_vars = "Name",
    var_name = "SampleID",
    value_name = "Input_value",
    score_name = "IGrad_value",
    palette = "Spectral",
    xlabel = "Integrated Gradient Scores",
    ylabel = "Lipids",
    title = "",
    draw_violin = TRUE,
    use_beeswarm = FALSE,
    ...
) {
    raw_data_df <- raw_data_df %>% tidyr::pivot_longer(cols = -{{id_vars}}, names_to = var_name, values_to = value_name)
    raw_data_df <- raw_data_df %>% dplyr::filter(!!dplyr::sym(var_name) %in% colnames(scores_df))

    tmp_df = scores_df %>% slice(1:num_biomols)
    mycols = colnames(tmp_df)[1:2]
    plot_df = tmp_df %>% tidyr::pivot_longer(cols = -mycols, names_to = var_name, values_to = score_name)

    plot_df = plot_df %>% dplyr::left_join(raw_data_df, by = c(setNames(id_vars, mycols[1]), var_name))

    # turn 'feature' into a factor with levels ordered by the shapley value:
    plot_df[[mycols[1]]] <- factor(plot_df[[mycols[1]]], levels = rev(dplyr::distinct(select(plot_df, one_of(mycols)))[[mycols[1]]]))

    plot_fn <- if(use_beeswarm) {
        if(!requireNamespace("ggbeeswarm")) stop("ggbeeswarm must be installed if use_beeswarm = TRUE")
        ggbeeswarm::geom_beeswarm
    } else {
        ggplot2::geom_jitter
    }

    p <- ggplot2::ggplot(plot_df, aes(x = !!dplyr::sym(score_name), y = !!dplyr::sym(mycols[1]), color = !!dplyr::sym(value_name)))

    if (draw_violin) {
        p <- p + ggplot2::geom_violin()
    }

    p <- p +
        plot_fn(...) + 
        ggplot2::geom_jitter() + 
        ggplot2::scale_color_viridis_c() + 
        ggplot2::theme_bw() + 
        ggplot2::labs(x = xlabel, y = ylabel, title = title)

    return(p)
}
