# Example of creating an importance plot from the results of the DeepIMV pipeline.
# Specifically, we take the raw e_data, as well as the shapley/IGrad scores, and make a beeswarm plot similar
# to the one in the SHAP python package.

library(readr)
library(ggplot2)
library(ggbeeswarm)
library(dplyr)
library(tidyr)

# load e_data and variable importance table from the DeepIMV pipeline
shap_df = read_csv("/Users/clab683/git_repos/DeepIMV/data/shapley_values_ICL104_proteins_luke.csv.csv")
#shap_df = read_csv("/Users/clab683/git_repos/DeepIMV/data/integrated_gradients_ICL104_proteins_luke.csv.csv")
#shap_df = read_csv("/Users/clab683/git_repos/DeepIMV/output/igrads-notebook-proteins-rangefix-2024-01-31.csv")
vals_df = read_csv("/Users/clab683/git_repos/DeepIMV/data/edata/ICL104_proteins_luke.csv")

# See the documentation in R/importance_plot.R
id_vars = "Protein"
var_name = "SampleID"
value_name = "Input_value"
score_name = "Shapley Value"
num_biomols = 10

p <- importance_plot(
    shap_df,
    vals_df,
    id_vars = id_vars, 
    var_name = var_name, 
    value_name = value_name, 
    score_name = score_name,
    xlabel = "Shapley Value",
    ylabel = "Protein"
)

p <- p + scale_y_discrete(limits = rev) + coord_flip()
p <- p + ggtitle("Shapley values for top proteins across all samples")
saveRDS(p, "shapley_proteins.rds")

shap_df = read_csv("/Users/clab683/git_repos/DeepIMV/data/shapley_values_ICL104_lipids_aligned_for_stats.csv.csv")
vals_df = read_csv("/Users/clab683/git_repos/DeepIMV/data/edata/ICL104_lipids_aligned_for_stats.csv")

id_vars = "Name"
var_name = "SampleID"
value_name = "Input_value"
score_name = "Shapley Value"
num_biomols = 10

p <- importance_plot(
    shap_df,
    vals_df,
    id_vars = id_vars, 
    var_name = var_name, 
    value_name = value_name, 
    score_name = score_name,
    xlabel = "Shapley Value",
    ylabel = "Lipid"
)

p <- p + scale_y_discrete(limits = rev) + coord_flip()
p <- p + ggtitle("Shapley values for top lipids across all samples")
# rotate x-axis labels
p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
saveRDS(p, "shapley_lipids.rds")


shap_df = read_csv("data/shapley_values_OMICS_ICL104_Metabolomics_YMK.csv.csv")
vals_df = read_csv("data/edata/OMICS_ICL104_Metabolomics_YMK.csv")

shap_df = shap_df %>% filter(!grepl("Unknown", feature))
vals_df = vals_df %>% filter(!grepl("Unknown", Metabolite)) %>% select(-one_of(c("KEGG", "CAS", "PubChem")))

id_vars = "Metabolite"
var_name = "SampleID"
value_name = "Input_value"
score_name = "Shapley Value"
num_biomols = 10

p <- importance_plot(
    shap_df,
    vals_df,
    id_vars = id_vars, 
    var_name = var_name, 
    value_name = value_name, 
    score_name = score_name,
    xlabel = "Shapley Value",
    ylabel = "Metabolite"
)

p <- p + scale_y_discrete(limits = rev) + coord_flip()
p <- p + ggtitle("Shapley values for top metabolites across all samples")
# rotate x-axis labels
# p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
saveRDS(p, "shapley_metabolites.rds")

# same for integrated gradients

shap_df = read_csv("data/output/integrated_gradients_ICL104_proteins_luke.csv.csv")
vals_df = read_csv("data/edata/ICL104_proteins_luke.csv")

id_vars = "Protein"
var_name = "SampleID"
value_name = "Input_value"
score_name = "IGrad_value"
num_biomols = 10

p <- importance_plot(
    shap_df,
    vals_df,
    id_vars = id_vars, 
    var_name = var_name, 
    value_name = value_name, 
    score_name = score_name,
    xlabel = "Integrated Gradient Scores",
    ylabel = "Protein"
)

p <- p + scale_y_discrete(limits = rev) + coord_flip()
p <- p + ggtitle("Integrated Gradient scores for top proteins across all samples")
saveRDS(p, "igrad_proteins.rds")

shap_df = read_csv("data/output/integrated_gradients_ICL104_lipids_aligned_for_stats.csv.csv")
vals_df = read_csv("data/edata/ICL104_lipids_aligned_for_stats.csv")

id_vars = "Name"
var_name = "SampleID"
value_name = "Input_value"
score_name = "IGrad_value"
num_biomols = 10

p <- importance_plot(
    shap_df,
    vals_df,
    id_vars = id_vars, 
    var_name = var_name, 
    value_name = value_name, 
    score_name = score_name,
    xlabel = "Integrated Gradient Scores",
    ylabel = "Lipid"
)

p <- p + scale_y_discrete(limits = rev) + coord_flip()
p <- p + ggtitle("Integrated Gradient scores for top lipids across all samples")
# rotate x-axis labels
p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
saveRDS(p, "igrad_lipids.rds")


shap_df = read_csv("data/output/integrated_gradients_OMICS_ICL104_Metabolomics_YMK.csv.csv")
vals_df = read_csv("data/edata/OMICS_ICL104_Metabolomics_YMK.csv")

shap_df = shap_df %>% filter(!grepl("Unknown", feature))
vals_df = vals_df %>% filter(!grepl("Unknown", Metabolite)) %>% select(-one_of(c("KEGG", "CAS", "PubChem")))

id_vars = "Metabolite"
var_name = "SampleID"
value_name = "Input_value"
score_name = "IGrad_value"
num_biomols = 10

p <- importance_plot(
    shap_df,
    vals_df,
    id_vars = id_vars, 
    var_name = var_name, 
    value_name = value_name, 
    score_name = score_name,
    xlabel = "Integrated Gradient Scores",
    ylabel = "Metabolite"
)

p <- p + scale_y_discrete(limits = rev) + coord_flip()
p <- p + ggtitle("Integrated Gradient scores for top metabolites across all samples")
# rotate x-axis labels
# p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
saveRDS(p, "igrad_metabolites.rds")

########

# vals_df <- vals_df %>% pivot_longer(cols = -{{id_vars}}, names_to = var_name, values_to = value_name)
# vals_df %>% pivot_longer(cols = -{{id_vars}}, names_to = var_name, values_to = value_name)

# vals_df <- vals_df %>% dplyr::filter(!!dplyr::sym(var_name) %in% colnames(shap_df))

# tmp_df = shap_df %>% slice(1:num_biomols)
# mycols = colnames(tmp_df)[1:2]
# igrad_plot_df = tmp_df %>% pivot_longer(cols = -mycols, names_to = var_name, values_to = score_name)

# igrad_plot_df = igrad_plot_df %>% left_join(vals_df, by = c(setNames(id_vars, mycols[1]), var_name))

# # turn 'feature' into a factor with levels ordered by the shapley value:
# igrad_plot_df$feature <- factor(igrad_plot_df$feature, levels = rev(distinct(select(igrad_plot_df, one_of(mycols)))$feature))

# # reverse the order of a vector

# # make a beeswarm plot in ggplot2
# ggplot(igrad_plot_df, aes(x = !!dplyr::sym(score_name), y = feature, color = {{value_name}})) + 
#     geom_violin() + 
#     geom_jitter() +
#     scale_color_viridis_c() + 
#     theme_bw() + 
#     theme(legend.position = "none") + 
#     labs(x = "Integrated Gradient Scores", y = "Protein")

# # place

# importance_plot <- function(
#     scores_df,
#     raw_data_df,
#     num_biomols = 10,
#     id_vars = "Name",
#     var_name = "SampleID",
#     value_name = "Input_value",
#     score_name = "IGrad_value",
#     palette = "Spectral",
#     xlabel = "Integrated Gradient Scores",
#     ylabel = "Lipids",
#     title = "",
#     draw_violin = TRUE,
#     use_beeswarm = FALSE,
#     coord_flip = TRUE,
#     ...
# ) {
#     raw_data_df <- raw_data_df %>% pivot_longer(cols = -{{id_vars}}, names_to = var_name, values_to = value_name)
#     raw_data_df <- raw_data_df %>% dplyr::filter(!!dplyr::sym(var_name) %in% colnames(scores_df))

#     tmp_df = scores_df %>% slice(1:num_biomols)
#     mycols = colnames(tmp_df)[1:2]
#     plot_df = tmp_df %>% pivot_longer(cols = -mycols, names_to = var_name, values_to = score_name)

#     plot_df = plot_df %>% left_join(raw_data_df, by = c(setNames(id_vars, mycols[1]), var_name))

#     # turn 'feature' into a factor with levels ordered by the shapley value:
#     plot_df[[mycols[1]]] <- factor(plot_df[[mycols[1]]], levels = rev(distinct(select(plot_df, one_of(mycols)))[[mycols[1]]]))

#     plot_fn <- if(use_beeswarm) {
#         if(!requireNamespace("ggbeeswarm")) stop("ggbeeswarm must be installed if use_beeswarm = TRUE")
#         ggbeeswarm::geom_beeswarm
#     } else {
#         ggplot2::geom_jitter
#     }

#     p <- ggplot(plot_df, aes(x = !!dplyr::sym(score_name), y = !!dplyr::sym(mycols[1]), color = !!dplyr::sym(value_name)))

#     if(draw_violin) {
#         p <- p + geom_violin()
#     }

#     p <- p +
#         plot_fn(...) + 
#         geom_jitter() + 
#         scale_color_viridis_c() + 
#         theme_bw() + 
#         labs(x = xlabel, y = ylabel, title = title)

#     if(coord_flip) {
#         p <- p + coord_flip()
#     }

#     return(p)
# }

# id_vars = "Protein"
# var_name = "SampleID"
# value_name = "Input_value"
# score_name = "Shapley Value"

# importance_plot(
#     shap_df,
#     vals_df,
#     id_vars = id_vars, 
#     var_name = var_name, 
#     value_name = value_name, 
#     score_name = score_name,
#     xlabel = "Shapley Value",
#     ylabel = "Protein",
#     coord_flip = FALSE
# )
