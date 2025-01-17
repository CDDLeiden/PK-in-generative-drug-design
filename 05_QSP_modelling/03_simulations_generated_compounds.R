###
### This script is used to simulate tumor growth with generated compounds
### QSPR model predicted properties of the generated compounds
### replace AZD4635 PK and A2AR binding affinity
###
library(rxode2)
library(ggplot2)
library(ggpubr)
library(data.table)
library(units)
library(dplyr)
library(tidyverse)
library(psych)
library(colorspace)

set.seed(42)
rxSetSeed(42)

# set current working directory to the location of this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("models/TME_model_1cmp.R")

# read in config.json file
config <- jsonlite::fromJSON("../config.json") # SET TO THE CORRECT PATH
figures_folder <- file.path(config$BASE_DIR, config$FIGURES_DIR, "QSP_modelling/")
figures_generated <- file.path(figures_folder, "simulations_generated_compounds/")

# create figures directory if it does not exist
if (!dir.exists(figures_folder)) dir.create(figures_folder)
if (!dir.exists(figures_generated)) dir.create(figures_generated)

# save package info to the figures folder
packages_info <- as.data.frame(do.call(rbind, lapply(sessionInfo()$otherPkgs, function(pkg) {
    data.frame(Package = pkg$Package, Version = pkg$Version)
})))
packages_info <- rbind(packages_info, data.frame(Package = "base", Version = R.version.string))
write.csv(packages_info, file.path(figures_generated, "session_info.csv"), row.names = FALSE)

a2arcolors <- c(
  "#FFCEAD", "#AAC5AC", "#447464", "#575463", "#9E949D",
  "#C46D5E", "#F4AC32",  "#7EB77F", "#20A39E"
)
a2arcolors_dark <- darken(a2arcolors, amount = 0.4)


####-------Read in properties of generated compounds-------####
generate_path <- file.path(config$BASE_DIR, config$PROCESSED_DATA_DIR, "DNDD/generated/")
generated_compounds <- list(
  AR = read.delim(file.path(generate_path, "A2AR_0/generated_10000.tsv")),
  AR_maxCL = read.delim(file.path(generate_path, "A2AR_CLmax_0/generated_10000.tsv")),
  AR_minCL = read.delim(file.path(generate_path, "A2AR_CLmin_0/generated_10000.tsv")),
  AR_maxVDSS = read.delim(file.path(generate_path, "A2AR_VDSSmax_0/generated_10000.tsv")),
  AR_minVDSS = read.delim(file.path(generate_path, "A2AR_VDSSmin_0/generated_10000.tsv"))
)

# create a boxplot facetted by parameter (VDSSmax_scorer, CLmax_scorer, A2AR_scorer)
# and color by scenario (not in manuscript)
merged_compounds <- rbindlist(generated_compounds, idcol = "scenario", fill = TRUE)
merged_compounds <- merged_compounds %>%
  select(
    scenario,
    FUmax_scorer,
    VDSSmax_scorer,
    CLmax_scorer,
    A2AR_scorer
  ) %>%
  gather(key = "parameter", value = "value", -scenario) %>%
  mutate(
    scenario = factor(
      scenario,
      levels = c(
        "AR",
        "AR_maxCL",
        "AR_minCL",
        "AR_maxVDSS",
        "AR_minVDSS"
      )
    )
  )
merged_compounds$parameter <- factor(
  merged_compounds$parameter,
  levels = c("FUmax_scorer", "VDSSmax_scorer", "CLmax_scorer", "A2AR_scorer"),
  labels = c("FU", "VDSS", "CL", "A2AR")
)
ggplot(merged_compounds, aes(y = value, fill = scenario)) +
  geom_boxplot() +
  facet_wrap(~parameter, scales = "free_y") +
  theme_bw(base_size = 15)
ggsave(
  file.path(figures_generated, "boxplot_generatedCompounds.png"),
  dpi = 300
)



####-------Define the model parameters-------####
# inter-individual variability
omega <- lotri(eta.sR ~ 0, eta.sL ~ 0)

sims_list <- list()
for (scenario in generated_compounds){
  # select only the compounds that are applicable all QSAR models and unique
  scenario <- scenario %>%
    filter(
      Unique...Valid...Applicable...Novel == 1,
    )

  ## define treatment groups
  # number of ids per treatment
  print(nrow(scenario))
  group_size <- nrow(scenario)

  # iv dosing in central compartment
  placebo <- et(id = 1:group_size) %>% # number of individuals
    et(seq(0, 30, 0.1)) # sampling times

  ARinh_dosing <- placebo %>%
    et(amt = 1.25, addl = 30, ii = 0.5, time = 7, cmt = "Ac2")

  combo <- placebo %>%
    et(amt = 0.125, addl = 4, ii = 3.5, time = 7, cmt = "Ad1") %>%
    et(amt = 1.25, addl = 30, ii = 0.5, time = 7, cmt = "Ac2")

  # Study specific value for sL from CIV151
  # individual covariates for MCA205 syngeneic model
  param_ind  <- expand.grid(
    id = 1:group_size,
    sL_cov = 0,
    TVin_cov = 0.69,
    Vado_cov = -3,
    sR_cov = 0.5308
  )

  # convert to Ki from pKi
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

  # create a data frame with the parameters
  sim_params <- data.frame(
    id = 1:group_size,
    Vc_ARinh = scenario$VDSS,
    kel_ARinh = scenario$kel,
    Kd_ARinh = scenario$A2AR,
    MW = scenario$MolWt
  )

  ## make model simulations with changed parameters
  placebo_sim <- rxSolve(
    TME_model_1cmp, sim_params, placebo, iCov = param_ind, omega = omega
  )
  ARinh_sim <- rxSolve(
    TME_model_1cmp, sim_params, ARinh_dosing, iCov = param_ind, omega = omega
  )
  combo_sim <- rxSolve(
    TME_model_1cmp, sim_params, combo, iCov = param_ind, omega = omega
  )

  # combine the simulations and add to the list
  sims <- rbindlist(
    list(
      placebo = placebo_sim,
      ARinh_dosing = ARinh_sim,
      combo = combo_sim
    ),
    idcol = "trt"
  )
  sims_list <- c(sims_list, list(sims))
}
names(sims_list) <- names(generated_compounds)
sims <- rbindlist(sims_list, idcol = "scenario")
rm("sims_list", "sim_params", "placebo_sim", "ARinh_sim", "combo_sim")
gc()

# make scenario column a factor
sims$scenario <- factor(
  sims$scenario,
  levels = c(
    "AR",
    "AR_maxCL",
    "AR_minCL",
    "AR_maxVDSS",
    "AR_minVDSS"
  )
)

####-------Plotting model simulations-------####

lb <- as_labeller(
  c(
    `combo` = "combination",
    `ARinh_dosing` = "ARinh_dosing4635",
    `placebo` = "placebo",
    `AR` = "A2AR",
    `AR_maxCL` = "A2AR + maximize CL",
    `AR_maxVDSS` = "A2AR + maximize VDSS",
    `AR_minCL` = "A2AR + minimize CL",
    `AR_minVDSS` = "A2AR + minimize VDSS"
  )
)

# Function to summarize statistics
summarize_stats <- function(data, variable) {
  data %>%
    group_by(time, trt, scenario) %>%
    summarise(
      n = n(),
      mean = mean(.data[[variable]]),
      median = median(.data[[variable]]),
      sd = sd(.data[[variable]]),
      min = min(.data[[variable]]),
      max = max(.data[[variable]])
    ) %>%
    mutate(
      se = sd / sqrt(n),
      CI_lower = mean - qnorm(0.975) * se,
      CI_upper = mean + qnorm(0.975) * se,
      PI_lower = mean - 1.96 * sd,
      PI_upper = mean + 1.96 * sd,
      variable = variable
    )
}

sims_mean <- summarize_stats(sims, "Tum")

# Plot tumor volume over time for all scenarios (Figure 8C)
TRT <- "combo"
ggplot(sims_mean %>% filter(trt == TRT), aes(time, mean)) +
  geom_ribbon( # plot the 95% interval of scenario max A2AR in the background
    data = sims_mean %>%
      filter(scenario == "AR", trt == TRT) %>%
      select(-scenario),
    aes(ymin = PI_lower, ymax = PI_upper),
    fill = "#b4b4b4",
    alpha = 0.5,
  ) +
  geom_ribbon( # plot the 95% prediction interval per scenario
    aes(ymin = PI_lower, ymax = PI_upper, fill = scenario),
    alpha = 0.9
  ) +
  geom_line( # plot the mean of the scenario max A2AR
    data = sims_mean %>%
      filter(scenario == "AR", trt == TRT) %>%
      select(-scenario),
    linetype = "longdash",
    color = "#313131",
  ) +
  geom_line(aes(x = time, y = mean, color = scenario), show.legend = FALSE) +
  geom_vline( # plot start of treatment
    xintercept = 7, linetype = "dashed", color = "#575463", linewidth = 1
  ) +
  geom_vline( # plot end of treatment
    xintercept = 22, linetype = "dashed", color = "#575463", linewidth = 1
  ) +
  facet_grid(~ scenario, labeller = lb, scales = "free_y") +
  labs(fill = "Scenario") +
  xlab("Time (days)") +
  ylab("Tumor volume (mm3)") + theme_light(base_size = 20) +
  scale_color_manual(values = a2arcolors_dark) +
  scale_fill_manual(values = a2arcolors, name = "Scenario", labels = lb) +
  theme(
    text = element_text(color = "#575463"),
    axis.text = element_text(color = "#575463"),
    legend.text = element_text(size = 15),
    strip.text.x = element_text(size = 15, color = "#575463", face = "bold"),
    strip.background = element_rect(colour = "#575463", fill = "#FFFFFF")
  )
ggsave(
  file.path(figures_generated, "simulation_of_Tum_generatedCompounds.png"),
  dpi = 600,
  width = 25,
  height = 6
)