library(rxode2)
library(ggplot2)
library(ggpubr)
library(data.table)
library(units)
library(dplyr)
library(tidyverse)
library(psych)
library(colorspace)
library("viridis")

setwd("Scripts/drugex-with-pk/QSP_modelling/")
source("models/TME_model_1cmp.R")
figures_folder <- "../../../Data/03_figures/QSP_modelling/"

a2arcolors <- c(
  "#FFCEAD", "#AAC5AC", "#447464", "#575463", "#9E949D",
  "#C46D5E", "#F4AC32",  "#7EB77F", "#20A39E"
)
a2arcolors_dark <- darken(a2arcolors, amount = 0.4)

set.seed(42)
rxSetSeed(42)

# set current working directory to the location of this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("models/TME_model_1cmp.R")

# read in config.json file
config <- jsonlite::fromJSON("../config.json") # set to the location of the config file
figures_folder <- file.path(config$BASE_DIR, config$FIGURES_DIR, "QSP_modelling/")
figures_example <- file.path(figures_folder, "example_simulations/")

# create figures directory if it does not exist
if (!dir.exists(figures_folder)) dir.create(figures_folder)
if (!dir.exists(figures_example)) dir.create(figures_example)

# save package info to the figures folder
packages_info <- as.data.frame(do.call(rbind, lapply(sessionInfo()$otherPkgs, function(pkg) {
  data.frame(Package = pkg$Package, Version = pkg$Version)
})))
packages_info <- rbind(packages_info, data.frame(Package = "base", Version = R.version.string))
write.csv(packages_info, file.path(figures_example, "session_info.csv"), row.names = FALSE)

####-------Read in properties of generated compounds-------####
generate_path <- file.path(config$BASE_DIR, config$PROCESSED_DATA_DIR, "DNDD/generated/")
generated_compounds <- list(
  AR = read.delim(file.path(generate_path, "A2AR_0/generated_10000.tsv")),
  AR_maxCL = read.delim(file.path(generate_path, "A2AR_CLmax_0/generated_10000.tsv")),
  AR_minCL = read.delim(file.path(generate_path, "A2AR_CLmin_0/generated_10000.tsv")),
  AR_maxVDSS = read.delim(file.path(generate_path, "A2AR_VDSSmax_0/generated_10000.tsv")),
  AR_minVDSS = read.delim(file.path(generate_path, "A2AR_VDSSmin_0/generated_10000.tsv"))
)

calculate_parameter_values <- function(scenario) {

  scenario <- scenario %>%
    filter(
      Unique...Valid...Applicable...Novel == 1,
    )

  # convert to Ki (nM) from pKi (-log10(M))
  scenario$A2AR <- (
    10**(-scenario$A2AR_scorer) * (10**9)
  )

  # "Generally, there is a good correlation between body weight
  # and volume of distribution among species. The exponents of
  # volume do not vary as widely as clearance and revolve around
  # 0.8 to 1.10." Mahmood 2007, https://doi.org/10.1016/j.addr.2007.05.015
  # I choose the average of 0.95 as exponent and 70 kg as human weight
  # convert to L from L/kg
  scenario$VDSS <- (10**(scenario$VDSSmax_scorer) * 70) / ((70 / 0.025)^0.95)

  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4181675/ :
  # allometric scaling of CL with allometry exponent fixed at 0.65
  # Cl_human =  Cl_animal x (BW_human/BW_animal)^0.65
  # Cl_mice (mL/min) = Cl_human (ml/min/kg) * BW_human (kg) /
  #                    (BW_mice/BW_human)^0.65
  # Cl_mice (L/day) = Cl_mice (mL/min) * 60 (min/h) * 24 (h/day) * 0.001 (L/mL)
  # Kel_mice (/day) = Cl_mice (L/day) / Vd_mice (L)
  cl_mice <- (10**(scenario$CLmax_scorer) * 70) / ((70 / 0.025)^0.65) # in mL/min
  cl_mice <- cl_mice * 60 * 24 * 0.001 # convert to L/day
  scenario$kel <- cl_mice / scenario$VDSS

  return(scenario)
}
generated_compounds <- lapply(generated_compounds, calculate_parameter_values)
generated_compounds <- rbindlist(generated_compounds, idcol = "scenario", fill = TRUE)

####-------Plot tumor growth for example compounds (Figure 8B)-------####
# select example compounds
low_kel_high_affinity <- generated_compounds %>%
  filter(quantile(A2AR, 0.1) >= A2AR) %>%
  select(SMILES, A2AR, VDSS, kel, MolWt) %>%
  arrange(kel) %>%
  head(1)

high_kel_high_affinity <- generated_compounds %>%
  filter(quantile(A2AR, 0.1) >= A2AR) %>%
  select(SMILES, A2AR, VDSS, kel, MolWt) %>%
  arrange(desc(kel)) %>%
  head(1)

low_kel_low_affinity <- generated_compounds %>%
  filter(quantile(A2AR, 0.9) <= A2AR) %>%
  select(SMILES, A2AR, VDSS, kel, MolWt) %>%
  arrange(kel) %>%
  head(1)

high_kel_low_affinity <- generated_compounds %>%
  filter(quantile(A2AR, 0.9) <= A2AR) %>%
  select(SMILES, A2AR, VDSS, kel, MolWt) %>%
  arrange(desc(kel)) %>%
  head(1)

group_size <- 4
# Drug-specific parameters
theta_1cmp <- data.frame(
  id = 1:group_size,
  kel_ARinh = c(high_kel_low_affinity$kel, high_kel_high_affinity$kel,
                low_kel_low_affinity$kel, low_kel_high_affinity$kel), # 1/day, Maximal AZD4635 elimination rate
  Vc_ARinh	= c(high_kel_low_affinity$VDSS, high_kel_high_affinity$VDSS,
                low_kel_low_affinity$VDSS, low_kel_high_affinity$VDSS), # L, Central volume of distribution for AZD4635
  Kd_ARinh  = c(high_kel_low_affinity$A2AR, high_kel_high_affinity$A2AR,
                low_kel_low_affinity$A2AR, low_kel_high_affinity$A2AR), # nM, affinity of azd for A2AR (In suppl. Kd_A2AR_AZD4635, listed as 0.013 nM!!)
  MW        = c(high_kel_low_affinity$MolWt, high_kel_high_affinity$MolWt,
                low_kel_low_affinity$MolWt, low_kel_high_affinity$MolWt) # g/mol, molecular weight of AZD4635
)

# Study-specific parameters
param_ind  <- expand.grid(
  id = 1:group_size,
  sL_cov = 0,
  TVin_cov = 0.69,
  Vado_cov = -3,
  sR_cov = 0.5308
)

# No inter-individual variability
omega <- lotri(eta.sR ~ 0, eta.sL ~ 0)

# Define the dosing regimen
placebo <- et(id = 1:group_size) %>%
  et(seq(0, 30, 0.1)) # sampling times

combo <- placebo %>%
  et(amt = 0.125, addl = 4, ii = 3.5, time = 7, cmt = "Ad1") %>%
  et(amt = 1.25, addl = 30, ii = 0.5, time = 7, cmt = "Ac2")

# Simulate the placebo and combo dosing regimens
placebo_sim <- rxSolve(
  TME_model_1cmp, theta_1cmp, placebo, iCov = param_ind, omega = omega
)
combo_sim <- rxSolve(
  TME_model_1cmp, theta_1cmp, combo, iCov = param_ind, omega = omega
)

all_sim <- rbindlist(
  list(
    placebo = placebo_sim,
    combination = combo_sim
  ),
  idcol = "trt"
)
all_sim$id <- factor(
  all_sim$id,
  levels = 1:group_size,
  labels = c(
    "High elimination rate, low affinity",
    "High elimination rate, high affinity",
    "Low elimination rate, low affinity",
    "Low elimination rate, high affinity"
  )
)
all_sim$id <- paste0(
  all_sim$id,
  "\n(kel: ",
  round(all_sim$kel_ARinh, 1),
  " 1/day, Ki: ",
  round(all_sim$Kd_ARinh, 1),
  " nM)"
)
all_sim$id <- factor(all_sim$id, levels = unique(all_sim$id))

ggplot(
  all_sim[trt == "combination"],
  aes(x = time, y = Tum, color = factor(id), group = factor(id), linetype = factor(id))
) +
  geom_line(lwd = 1.5) +
  ylab("Tumor volume (mm3)") + theme_light(base_size = 20) +
  xlab("Time (days)") +
  scale_color_manual(values = a2arcolors) +
  scale_linetype_manual(values = c("solid", "dotted", "dashed", "dotdash")) +
  theme(legend.position = "bottom") +
  guides(
    fill = guide_legend(keywidth = 1, keyheight = 1),
    linetype = guide_legend(keywidth = 2, keyheight = 1, nrow = 2),
    colour = guide_legend(keywidth = 2, keyheight = 1, nrow = 2)
  ) +
  geom_vline( # plot start of treatment
    xintercept = 7, linetype = "dashed", color = "#575463", linewidth = 1
  ) +
  geom_vline( # plot end of treatment
    xintercept = 22, linetype = "dashed", color = "#575463", linewidth = 1
  ) +
  theme(
    legend.title = element_blank(),
    text = element_text(color = "#575463"),
    axis.text = element_text(color = "#575463"),
    legend.text = element_text(size = 15),
    strip.text.x = element_text(size = 15, color = "#575463", face = "bold"),
    strip.background = element_rect(colour = "#575463", fill = "#FFFFFF")
  )
ggsave(
  file.path(figures_example, "example_compounds.png"),
  width = 9,
  height = 8,
  dpi = 300
)
