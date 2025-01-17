###
### This script is a replication of the original code of Voronova et al., 2021 using RxODE
### DOI: https://doi.org/10.3389/fimmu.2021.617316
### Code: https://github.com/VeronikaVor2/IO-QSP-model-adenosine-
###

library(rxode2)
library(ggplot2)
library(ggpubr)
library(data.table)
library(dplyr)
library(tidyverse)
library(cowplot)
library(gridExtra)
library(psych)

# Note. Some figures show deviations from the original publication
# due to the random number generation.
set.seed(42)
rxSetSeed(42)

# set current working directory to the location of this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("models/TME_model_voronova_rewrite.R")

# read in config.json file
config <- jsonlite::fromJSON("../config.json") # set to the location of the config file
figures_folder <- file.path(config$BASE_DIR, config$FIGURES_DIR, "QSP_modelling/")
figures_rewrite_folder <- file.path(figures_folder, "figures_rewrite/")

# Read in flow cytometry data downloaded from the supplementary material of Voronova et al.
flow_cytometry_data <- file.path(
  config$BASE_DIR, config$RAW_DATA_DIR, "Voronova_data/FlowValidation_AZpaper.csv"
)

# create figures directory if it does not exist
if (!dir.exists(figures_folder)) dir.create(figures_folder)
if (!dir.exists(figures_rewrite_folder)) dir.create(figures_rewrite_folder)

# save package info to the figures folder
packages_info <- as.data.frame(do.call(rbind, lapply(sessionInfo()$otherPkgs, function(pkg) {
    data.frame(Package = pkg$Package, Version = pkg$Version)
})))
packages_info <- rbind(packages_info, data.frame(Package = "base", Version = R.version.string))
write.csv(packages_info, file.path(figures_rewrite_folder, "session_info.csv"), row.names = FALSE)

# AZD population PK parameters (other pop parameters in RxODE TME_model_rewrite)
theta <- c(
  Vmaxabs_azd = 12.6,   # mg/day, Maximal AZD4635 absorption rate
  EC50abs_azd = 0.178,  # mg, AZD amount to achieve 50% of max absorption rate
  Q_azd	      = 9.58,   # L/day, Intercompartmental clearance for AZD4635
  kel_azd     = 320,    # 1/day, Maximal AZD4635 elimination rate
  Vc_azd	    = 0.0476, # L, Central volume of distribution for AZD4635
  Vp_azd	    = 1.43,   # L, Peripheral volume of distribution for AZD4635
  Kd_azd      = 13.43   # nM, affinity of azd for A2AR (In suppl. Kd_A2AR_AZD4635, listed as 0.013 nM!!)
)

# inter-individual variability
omega <- lotri(eta.sR ~ 0.136836082237707^2, eta.sL ~ 0.322226249102547^2)
# omega <- lotri(eta.sR ~ 0, eta.sL ~ 0) # run without inter-individual variability

## define treatment groups
# number of ids per group (the four different studies)
group_size <- 30

placebo <- et(id = 1:(group_size * 4)) %>% # number of individuals
  et(seq(0, 30, 0.1)) # sampling times

# Note. a few corrections were made to the dosing to match Voronova et al., see comments below.
# 5 mg/kg dose of PD-L1 mAb every 3.5 days (BIW)
# 5 mg/kg * 0.025 kg bodyweight = 0.125 mg
PDL1 <- placebo %>% # sampling times
  et(amt = 2 * 6.66, addl = 4, ii = 3.5, time = 7, cmt = "Ad1") %>% # 2*6.66 = 13.2 in Voronova et al. (Now corrected in GitHub, effect on simulations is minimal)
  et(amt = 2 * 6.66, addl = 4, ii = 3.5, time = 7, cmt = "Ac1") # extra dose added to match Voronova et al.

# 50 mg/kg dose of AZD4635 every 12 hours (BID)
# 50 mg/kg * 0.025 kg bodyweight = 1.25 mg
AZD <- placebo %>% # sampling times
  et(amt = 2 * 1.25, addl = 30, ii = 0.5, time = 7, cmt = "Ad2") # double concentration to equal Voronova et al.

combo <- placebo %>% # sampling times
  et(amt = 2 * 6.66, addl = 4, ii = 3.5, time = 7, cmt = "Ad1") %>%
  et(amt = 2 * 1.25, addl = 30, ii = 0.5, time = 7, cmt = "Ad2") %>% # double concentration to equal Voronova et al.
  et(amt = 2 * 6.66, addl = 4, ii = 3.5, time = 7, cmt = "Ac1") # extra dose added to match Voronova et al.

# unadapted dosing
# PDL1 <- placebo %>% # sampling times
#  et(amt = 0.125, addl = 4, ii = 3.5, time = 7, cmt = "Ad1")

# AZD <- placebo %>% #sampling times
#  et(amt = 1.25, addl = 30, ii = 0.5, time = 7, cmt = "Ad2")

# combo <- placebo %>% #sampling times
#  et(amt = 0.125, addl = 4, ii = 3.5, time = 7, cmt = "Ad1") %>%
#  et(amt = 1.25, addl = 30, ii = 0.5, time = 7, cmt = "Ad2")

# Study specific value for sL
study <- array(
  data = c(0, -0.737482483434612, -0.747030668928987, 0.261260213403196),
  dim = c(4, 1),
  dimnames = list(c("CIV151", "CIV226", "CIV227", "CIV258"), c("sL"))
)

#Model specific value for TVin, Vado en sR
model <- array(
  data = c(0, 0, 0.69,
           0, -0.5, -3,
           0, -0.155518400669642, 0.530815811383967),
  dim = c(3, 3),
  dimnames = list(c("CT26", "MC38", "MCA205"),
                  c("TVin", "Vado", "sR"))
)

#---------------------------- model-based simulations of TME (figure 2) ------------------------------------------------

### define covariates
study_group <- data.frame(
  Study = factor(
    c("CIV226", "CIV227", "CIV258", "CIV151"),
    levels = c("CIV226", "CIV227", "CIV258", "CIV151"),
    ordered = TRUE
  ),
  Model = factor(
    c("CT26", "MC38", "MCA205", "MCA205"),
    levels = c("CT26", "MC38", "MCA205")
  ),
  ordered = TRUE
)

## Create a dataframe with the individual study/model specific parameter values for the covariates
param_ind <- expand.grid(id = 1:group_size, Study = study_group$Study) %>%
  merge(., study_group[, c("Study", "Model")], by = "Study") %>%
  mutate(id = seq_len(nrow(.)))

param_ind$sL_cov   <- study[as.character(param_ind$Study), "sL"]
param_ind$TVin_cov <- model[as.character(param_ind$Model), "TVin"]
param_ind$Vado_cov <- model[as.character(param_ind$Model), "Vado"]
param_ind$sR_cov   <- model[as.character(param_ind$Model), "sR"]

#Simulations for the different treatment groups
placebo_sim <- rxSolve(TME_model_rewrite, theta, placebo, iCov = param_ind, omega = omega, keep = c("Model", "Study"))
PDL1_sim    <- rxSolve(TME_model_rewrite, theta, PDL1, iCov = param_ind, omega = omega, keep = c("Model", "Study"))
AZD_sim     <- rxSolve(TME_model_rewrite, theta, AZD, iCov = param_ind, omega = omega, keep = c("Model", "Study"))
combo_sim   <- rxSolve(TME_model_rewrite, theta, combo, iCov = param_ind, omega = omega, keep = c("Model", "Study"))

# combine all the simulations
all_sim <- rbindlist(list(Vehicle = placebo_sim, PD_L1 = PDL1_sim, AZD=AZD_sim, combination=combo_sim), idcol = "trt")

# pivot to long format for easier plotting and rename values for study and model because they dissapear in rxsolve
all_sim <- all_sim %>%
  tidyr::pivot_longer(!c(id, time, trt, Study, Model), names_to = "var", values_to = "value") %>%
  mutate(
    trt = forcats::fct_relevel(trt, "Vehicle", "AZD", "PD_L1", "combination"),
    Study = recode(as.factor(Study), "1" = "CIV226", "2" = "CIV227", "3" = "CIV258", "4" = "CIV151"),
    Model = recode(as.factor(Model), "1" = "CT26", "2" = "MC38", "3" = "MCA205")
  )

# calculate quantiles for each variable to plot confidence intervals
quantiles <- all_sim %>%
  group_by(time, trt, Study, Model, var) %>%
  summarize(ci05 = quantile(value, 0.05),
            ci50 = quantile(value, 0.5),
            ci95 = quantile(value, 0.95)) %>%
  ungroup() %>%
  right_join(all_sim)

#plot confidence intervals for all four simulations per study
alt_col <- c("black", "green3", "blue", "red")
quantiles_subset <- quantiles[quantiles$var %in% c("Tum", "Ado_suppr", "PDL1free", "ISC", "CTL"), ]#), ]
ggplot(quantiles_subset) +
  facet_grid(as.factor(var) ~ Study, scales = "free") +
  geom_rect(data = quantiles_subset %>% group_by(var) %>% summarize(maxv = max(ci95)),
    aes(xmin = 7, xmax = 22, ymin = 0, ymax = maxv), alpha = 0.2) +
  geom_ribbon(aes(x=time, y=ci50, ymin=ci05, ymax=ci95, group=trt, fill=trt), alpha=0.5) +
  geom_line(aes(x=time, y=ci50, group=trt, color=trt, linetype = trt)) + theme_bw(base_size=12) +
  # geom_line(aes(x=time, y=value, group=interaction(id, trt), color=trt)) + theme_bw(base_size=12) +
  ylab("") +
  scale_color_manual(values = alt_col) + scale_fill_manual(values = alt_col) +
  scale_linetype_manual(values = c("solid", "dashed", "solid", "dashed")) +
  theme(plot.title = element_text(size = 16))
ggsave(paste0(figures_rewrite_folder, "/f2_abc.png"))

#---------------------------- evaluation of statistical differences between the groups on the day 30 (figure s4) -------

# define groups to t-test the tumor volume on day 30
my_comparisons <- list(c("PD_L1", "combination"), c("AZD", "combination"))
lb <- as_labeller(
  c(`CIV258` = "CIV258 \n MCA205: 2",
    `CIV226` = "CIV226 \n CT26",
    `CIV151` = "CIV151 \n MCA205",
    `CIV227` = "CIV227 \n MC38")
)

#boxplot of tumor volume for each simulated individual on day 30 of the simulations
ggplot(all_sim %>% filter(time == 30 & var == "Tum"), aes(x = trt, y = value)) +
  facet_wrap(~ Study, labeller = lb) +
  geom_boxplot(outlier.alpha = 0.001) +
  geom_jitter(width = 0.1, height = 0) +
  xlab("") + ylab(expression(Tumor~volume~mm^{3})) +
  scale_x_discrete(labels = c("Vehicle", "AZD4635", "PD-L1 mAb", "Combo\n50 mg/kg")) +
  theme_bw(base_size = 16) +
  stat_compare_means(comparisons = my_comparisons, method = "t.test", alternative = "two.sided") + ylim(0, 3500)
ggsave(paste0(figures_rewrite_folder, "/fs4.png"))

#---------------------------- validation using CFM data (figure s5) ----------------------------------------------------
# find the fold change of CTL and APC values from placebo treatment on day 23 of the simulations
valid_model <- all_sim %>%
  filter(time == 23 & var %in% c("Ag_norm", "CTL") & Study == "CIV227") %>%
  group_by(var) %>%
  mutate(median_pbo = median(value[trt == "Vehicle"])) %>%
  ungroup() %>%
  mutate(
    fold = value / median_pbo,
    process = recode(var, `Ag_norm` = "APC", `CTL` = "CTL")
  ) %>%
  filter(trt %in% c("Vehicle", "PD_L1", "AZD", "combination")) %>%
  mutate(
    Group = recode(
      trt,
      "Vehicle" = "Vehicle",
      "PD_L1" = "PD1Ab",
      "AZD" = "AZD4635",
      "combination" = "BID50"
    ),
    Variable = "model"
  ) %>%
  select(Model, Variable, Group, fold, process)

# Read in flow cytometry data and also calculate the fold change for this data
valid_data <- read.csv(flow_cytometry_data)
valid_data <- valid_data %>%
  group_by(Variable) %>%
  mutate(mean_pbo = median(Value[Group == "Vehicle"])) %>%
  mutate(fold = Value / mean_pbo) %>%
  filter(Variable != "CD8_perc") %>%
  mutate(
    process = recode(
      Variable,
      `CD86_DC` = "APC",
      `MHCII_DC` = "APC",
      `CD86_M` = "APC",
      `MHCII_M` = "APC",
      `CD8_PD1_perc` = "CTL"
    )
  ) %>%
  select(Model, Variable, Group, fold, process)

validation <- rbind(as.data.frame(valid_model), as.data.frame(valid_data))
validation$Group <- factor(
  validation$Group, levels = c("Vehicle", "PD1Ab", "AZD4635", "BID50")
)

# plot the flow cytometry data and simulated data fold changes in a boxplot
ggplot(validation) +
  facet_wrap(~process, scales = "free", labeller = as_labeller(c("CTL" = "CTL activation", "APC" = "APC activation"))) +
  geom_boxplot(aes(x = Group, fill = Variable, y = fold)) +
  theme_bw(base_size = 12) +
  ylab("vehicle-normalized value") + xlab("") +
  scale_fill_manual(
    values = c("salmon", "lightpink", "salmon", "red", "orange1",
               "lightblue", "lightblue"),
    labels = c("PD1+CD8 cells, % from live (data)", "CD86 on DC (MFI) (data)",
               "CD86 on M (MFI) (data)", "MHC II on DC (MFI) (data)",
               "MHC II on M (MFI) (data)", "dTeff or APC (model)")) +
  scale_x_discrete(labels = c("Vehicle", "PD-L1 Ab", "AZD4635", "Combo")) +
  ggtitle("Model validation") +
  theme(plot.title = element_text(size = 16))
ggsave(paste0(figures_rewrite_folder, "/fs5.png"))

#---------------------------- evaluation of between-animal variability (fig 3) -----------------------------------------
# simulate for 100 indviduals with study CIV258
param_ind <- expand.grid(id = 1:100, Study = "CIV258") %>%
  merge(., study_group[, c("Study", "Model")], by = "Study") %>%
  mutate(id = seq_len(nrow(.)))

param_ind$sL_cov   <- study[as.character(param_ind$Study), "sL"]
param_ind$TVin_cov <- model[as.character(param_ind$Model), "TVin"]
param_ind$Vado_cov <- model[as.character(param_ind$Model), "Vado"]
param_ind$sR_cov   <- model[as.character(param_ind$Model), "sR"]

placebo_sim <- rxSolve(TME_model_rewrite, theta, placebo, iCov = param_ind, omega = omega, nSub=100)
combo_sim <- rxSolve(TME_model_rewrite, theta, combo, iCov = param_ind, omega = omega, nSub=100)

# combine all the simulations
all_sim <- rbindlist(list(placebo = placebo_sim, combo=combo_sim), idcol = "trt")

# pivot to long format for easier plotting
# and rename values for study and model because they dissapear in rxsolve
all_sim <- all_sim %>%
  tidyr::pivot_longer(
    !c(id, time, trt), names_to = "var", values_to = "value"
  ) %>%
  mutate(trt = forcats::fct_relevel(trt, "placebo", "combo"))

## define responders and non-responders
resp <- all_sim %>%
  mutate(time = time) %>%
  filter(var == "Tum", time == 30 | time == 7) %>%
  pivot_wider(names_from = time, values_from = value) %>%
  rename("start" = "7", "end" =  "30") %>%
  mutate(resp = case_when(trt == "placebo" ~ "vehicle",
                          start > end ~ "resp",
                          TRUE ~ "prog")) %>%
  select(id, resp, trt)
resp_all_sim <- merge(all_sim, resp, by = c("id", "trt"))

## merge all data
resp_ci <- resp_all_sim %>%
  group_by(time, resp, var) %>%
  summarize(
    ci05 = quantile(value, 0.05),
    ci50 = quantile(value, 0.5),
    ci95 = quantile(value, 0.95)
  ) %>%
  ungroup() %>%
  mutate(resp = factor(resp, levels = c("vehicle", "resp", "prog"))) %>%
  filter(
    var %in% c("Tum", "ISC", "CTL", "Ado_suppr", "PDL1", "Ag_norm", "CD8_tot")
  )

## make TME "snapshots"
# select data a timepoint 14 and 7
resp_snapshots <- resp_all_sim %>%
  mutate(time = time) %>%
  filter(
    var %in% c("Tum", "ISC", "CTL", "Ado_suppr", "PDL1", "CD8_tot", "Ag_norm"),
    (time == 14) | (time == 7)
  ) %>%
  mutate(resp = factor(resp, levels = c("vehicle", "prog", "resp")))

# select sL_out and sR_out to plot density difference between responders and non-responders
param_resp <- resp_all_sim %>%
  select(id, var, value, resp) %>%
  filter(var %in% c("sL_out", "sR_out")) %>%
  unique() %>%
  filter(resp != "vehicle") %>%
  mutate(resp = factor(resp, levels = c("resp", "prog")))

# plot tumor dynamics for responders and non-responders separately
ggplot(resp_ci[resp_ci$var == "Tum", ]) +
  geom_line(aes(x = time, y = ci50, group = resp, color = resp)) +
  theme_bw(base_size = 12) +
  geom_ribbon(
    aes(x = time, ymin = ci05, ymax = ci95, group = resp, fill = resp),
    alpha = 0.2
  ) +
  xlab("Time (days)") + ylab("Tumor volume (uL)") +
  scale_colour_manual(
    values = c("blue", "forestgreen", "darkred"),
    name = "Group", labels = c("vehicle", "non-progressors", "progressors")
  ) +
  scale_fill_manual(
    values = c("blue", "forestgreen", "darkred"),
    name = "Group", labels = c("vehicle", "non-progressors", "progressors")
  ) +
  ggtitle("A.Tumor  dynamics") +
  theme(plot.title = element_text(size = 16), legend.position = "right")
ggsave(paste0(figures_rewrite_folder, "/f3a.png"))

# plot densitiy of sL_out (t effector cell flux) and immuno suppresive cells between responders and non-responders
ggplot(param_resp) +
facet_wrap(~var, scales="free", label=as_labeller(c("sL_out"="nTeff flux", "sR_out"="ISC flux"))) +
  geom_density(aes(x = 1/value, fill = resp), position = "identity", alpha = 0.5) +
  theme_bw(base_size = 12) +
  ggtitle("B.Estimated fluxes") +
  xlab("") +
  theme(plot.title = element_text(size = 16), legend.position = "none") +
  scale_fill_manual(values=c("forestgreen", "darkred"), name="Group", labels=c("non-progressors", "progressors"))
ggsave(paste0(figures_rewrite_folder, "/f3b.png"))

# plot difference in biomarker dynamics for responders and non-responders
labs_1 <- as_labeller(c(`CTL`="dTeff cells/uL", `ISC`="ISC", `CD8_tot`="Total CD8 cells uL", `PDL1`="total PD-(L)1"))

ggplot(resp_ci[(resp_ci$var%in% c("CTL", "ISC", "PDL1", "CD8_tot")), ]) +
  facet_wrap(~var, scales="free", labeller=labs_1, ncol=4) +
  geom_line(aes(x=time, y=ci50, group=resp, color=resp)) +
  theme_bw(base_size=12) +
  geom_ribbon(aes(x=time, ymin=ci05, ymax=ci95, group=resp, fill=resp), alpha=0.2) +
  xlab("Time") + ylab("value") +
  scale_colour_manual(values=c("blue", "forestgreen", "darkred"),
                      name="Group", labels=c("vehicle", "non-progressors", "progressors")) +
  scale_fill_manual(values=c("blue", "forestgreen", "darkred"),
                    name="Group", labels=c("vehicle", "non-progressors", "progressors")) +
  ggtitle("C.Biomarker dynamics") +
  theme(plot.title = element_text(size=16), legend.position = "none")
ggsave(paste0(figures_rewrite_folder, "/f3c.png"))

# boxplot of biomarker levels before and after treatment
ggplot(resp_snapshots[(resp_snapshots$var%in% c("CTL", "ISC", "PDL1", "CD8_tot")), ])+
  facet_wrap(~var, scales="free", labeller=labs_1, ncol=4)+
  geom_boxplot(aes(x=as.factor(time), y=value, fill=resp), alpha=0.4) +
  theme_bw(base_size=12) +
  scale_fill_manual(values=c("blue", "darkred", "forestgreen"),
                           name="Group", labels=c("vehicle", "progressors", "non-progressors"))+
  scale_x_discrete(labels=c("baseline", "post-treat")) +
  xlab("") +
  ggtitle("D.Biomarker levels") +
  theme(plot.title = element_text(size=16), legend.position = "none")
ggsave(paste0(figures_rewrite_folder, "/f3d.png"))

#---------------------------- sensitivity analysis (figure 4A) ---------------------------------------------------------
# create event table with 3 id's for default parameter, x2 and x0.5
param_ind <- expand.grid(id = 1:3, Study = "CIV227", vacc=1) %>% # set vacc to 1 to match voronova, et al.
             merge(., study_group[, c("Study", "Model")], by = "Study") %>%
             mutate(id = seq_len(nrow(.)))

param_ind$sL_cov   <- study[as.character(param_ind$Study), "sL"]
param_ind$TVin_cov <- model[as.character(param_ind$Model), "TVin"]
param_ind$Vado_cov <- model[as.character(param_ind$Model), "Vado"]
param_ind$sR_cov   <- model[as.character(param_ind$Model), "sR"]

### define parameters to test
par_var <- c("TsL", "TsR", "kLn", "TVado", "Kp", "beff", "r", "Ag_fl")

default_params <- coef(TME_model_rewrite, complete = FALSE)$ini[par_var]

omega <- lotri(eta.sR ~ 0, eta.sL ~ 0)

scF <- 2

## make model simulations with changed parameters
sensitivity <- as.data.frame(matrix(ncol=10, nrow=0))
for (par in par_var) {
  sim_params <- data.frame(id = 1:3, par = c(default_params[par], default_params[par] / scF,
                                             default_params[par] * scF))
  names(sim_params)[2] <- par
  sim_params <- bind_cols(sim_params, data.frame(t(replicate(3, theta))))

  AZD_sim   <- rxSolve(TME_model_rewrite, sim_params[-1], AZD, iCov = param_ind, omega = omega, nSub=3)
  combo_sim <- rxSolve(TME_model_rewrite, sim_params[-1], combo, iCov = param_ind, omega = omega, nSub=3)

  sims <- rbindlist(list(mono = AZD_sim, combo = combo_sim), idcol = "trt")

  colnames(sim_params)[which(names(sim_params) == par)] <- "value_sim_par"

  sims <- sims %>%
          pivot_longer(!c(id, time, trt), names_to = "var", values_to = "value") %>%
          left_join(y = sim_params[c("id", "value_sim_par")], by = "id") %>%
          # filter(var== "Tum", time==30, trt == "combo") %>%
          group_by(trt, time, var) %>%
          mutate(sim_par = par,
                 id = recode(id, "1"="default", "2"="x0.5", "3"="x2")) %>%
          mutate(rel_change = (value-value[id=="default"])/value[id=="default"]*100) %>%
          mutate(rel_change = replace_na(rel_change, 0)) %>%
          ungroup()

  colnames(sensitivity) <- colnames(sims)
  sensitivity <- rbind(sensitivity, sims)
}

# select tumor volume on time point 30 and order the dataset
sensitivity <- sensitivity %>%
               filter(var== "Tum", time==30, id != "default") %>%
               arrange(trt) %>%
               mutate(trt=as.factor(recode(trt, "mono"="AZD4635 alone", "combo"="AZD4635 + anti-PD-L1 mAb")),
               sim_par=fct_reorder(sim_par, rel_change, .desc=TRUE),
               trt=fct_relevel(trt, "AZD4635 alone", "AZD4635 + anti-PD-L1 mAb"))

# plot the change in tumor volume for different parameter values
ggplot(sensitivity, aes(sim_par, rel_change, fill=as.factor(id))) +
  facet_wrap(~trt) +
  theme_bw(base_size=12) +
  geom_bar(position="identity", stat="identity", alpha=0.5, color="black") +
  coord_flip() +
  scale_x_discrete(labels=c("kLn"="kLn", "beff"="beff", "r"="r", "TsL"= "sL", "TsR"="sR",
                            "Ag_fl"= expression("Ag"["norm"]), "TVado"=expression("Vmax"["Ado"]), "Kp"="Kp")) +
  xlab("parameter") + ylab("Tumor volume change from the default simulation on day 30, %") +
  scale_fill_manual(values=c("red", "blue"), name="parameter\nchange")
ggsave(paste0(figures_rewrite_folder, "/f4a.png"))

#---------------------------- alternative treatment combinations (figure 4B) -------------------------------------------
## create a dataframe, specifying parameters for various treatment options
group_size <- 10

param_ind <- data.frame(id=1:(5*group_size), Study="CIV227", Model="MC38",
                        cd73=rep(c(0, 1, 0, 0, 0), each=group_size), act=rep(c(0, 0, 1, 0, 0), each=group_size),
                        cytost=rep(c(0, 0, 0, 1, 0), each=group_size), vacc=rep(c(0, 0, 0, 0, 1), each=group_size))

param_ind$sL_cov   <- study[as.character(param_ind$Study), "sL"]
param_ind$TVin_cov <- model[as.character(param_ind$Model), "TVin"]
param_ind$Vado_cov <- model[as.character(param_ind$Model), "Vado"]
param_ind$sR_cov   <- model[as.character(param_ind$Model), "sR"]

omega <- lotri(eta.sR ~ 0.136836082237707^2, eta.sL ~ 0.322226249102547^2)

## run 100 model simulations with 10 animals per group
placebo_sim <- rxSolve(TME_model_rewrite, theta, placebo, iCov = param_ind, omega = omega, nSub=group_size*5, nStud = 100)
AZD_sim     <- rxSolve(TME_model_rewrite, theta, AZD, iCov = param_ind, omega = omega, nSub=group_size*5, nStud = 100)
combo_sim   <- rxSolve(TME_model_rewrite, theta, combo, iCov = param_ind, omega = omega, nSub=group_size*5, nStud = 100)

sims <- rbindlist(list(plac = placebo_sim, mono = AZD_sim, combo = combo_sim), idcol = "trt")

## process the simulations
# assign treatment to id numbers
sims <- sims %>%
        mutate(addtrt=ifelse(id %in% 1:10, "no",
                      ifelse(id %in% 11:20, "cd73",
                      ifelse(id %in% 21:30, "act",
                      (ifelse(id %in% 31:40, "cytost", "vacc"))))))

# take TGI as geometric mean of tumor volume normalized by the placebo treatment after timepoint 27.9
sims <- sims %>%
        pivot_longer(!c(id, sim.id, time, trt,  addtrt), names_to = "var", values_to = "value") %>%
        filter(var=="Tum" & time>27.9) %>%
        group_by(time, trt, var, sim.id, addtrt) %>%
        mutate(TV_m=geometric.mean(value)) %>%
        group_by(sim.id, var, addtrt) %>%
        mutate(TGI= (1-TV_m/TV_m[trt=="plac"])*100) %>%
        group_by(trt, var, addtrt) %>%
        summarize(mean_tg=median(TGI), ci05=quantile(TGI, 0.05), ci95=quantile(TGI, 0.95)) %>%
        ungroup() %>%
        mutate(addtrt=fct_relevel(addtrt, "act", "cytost", "cd73", "vacc", "no"),
                                  trt=fct_relevel(trt, "plac", "mono", "combo"))

## visualise the tumor growth inhibition for different treatments
ggplot(sims %>% filter(trt!="plac")) +
       geom_tile(aes(x=trt, y=addtrt, fill=mean_tg), color="black") +
       theme_bw(base_size=16) +
       scale_fill_gradientn(colours=c("red3", "yellow", "green", "forestgreen"),
                            limits=c(0, 100), name="TGI, %") +
       geom_text(aes(x = trt, y = addtrt,
                label =  paste(round(mean_tg), "\n (", round(ci05), "-", round(ci95), ")", sep="")),
                size=5, color="black") +
  scale_y_discrete(labels=c("no"="+ vehicle", "vacc"="+vaccine", "cd73"="+CD73 Ab",
                             "cytost"="+ ISC bl.", "act"="+ACT")) +
  scale_x_discrete(labels=c("mono"="AZD4635 \n alone", "combo"="AZD4635+ \n anti-PD-L1 mAb")) +
  xlab("") + ylab("")
ggsave(paste0(figures_rewrite_folder, "/f4b.png"))
