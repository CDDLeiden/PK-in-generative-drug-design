TME_model_1cmp <- RxODE({
  ## theta
  TVmax  <- 3500;              # uL, Maximal size of tumor # nolint
  beff   <- 0.001;             # 1/(d*cell), Rate of tumor cell kill by dTeff (In suppl. bef)
  r      <- 0.521743095907443; # 1/d, Tumor growth rate
  kLn    <- 209.674023531827;  # cells/d, Maximal influx rate of nTeff cells
  TsL    <- 4.55575391256157;  # uL/d, Tcell infiltration tumor under Ag exposure
  Kp     <- 1279.91714283438;  # cells, Sensitivity of PD-L1 expression up-regulation to dTeff count (In suppl. 478)
  Kado   <- 80;                # 1/d, Adenosine accumulation rate constant
  IC50   <- 1.82315256991075;  # unitless, the A2AR occupancy with a 50% decrease in immune cell activity (In suppl eq 16 EC50_A2ARs, not in suppl params table) 
  TsR    <- 57.0519720073586;  # uL/d, Sensitivity of cellular immunosuppression
  sCf    <- 1;                 # if 1 adenosine conc at the site of action is ~ measured intratumoral level,
                               # a scaling factor enabling conversion of total intratumoral adenosine content to extracellular adenosine levels (In suppl. Ado_sCf)
  TVado  <- 100000;            # nM, adenosine level in tumor / reflecting the adenosine synthesis rate (In suppl. Vmax_ado, listed as 100 uM)
  TTVin  <- 2.05452388904701;  # uL, Initial tumor volume (In suppl. TV0)

  ## theta_1cmp a2ar inhibitors
  Vc_ARinh <- Vc_ARinh;
  Kd_ARinh <- Kd_ARinh;
  kel_ARinh <- kel_ARinh; # elimination rate of AZD4635
  MW <- MW; # molecular weight

  ## variables with covariates and/or variance##
  sL <- TsL * exp(eta.sL + sL_cov);  # uL/d, T cell ability to infiltrate tumor tissue under systemic antigen exposure
  sR <- TsR * exp(eta.sR + sR_cov);  # uL/d, Sensitivity of cellular immunosuppression
  Vado <- TVado * exp(Vado_cov);     # nM, adenosine level in tumor / reflecting the adenosine synthesis rate (In suppl. Vmax_ado, listed as 100 uM)
  TVin <- TTVin * exp(TVin_cov);     # uL, Initial tumor volume (In suppl. TV0)

  # PD-L1 monoclonal antibody
  convF1   <- 6.66;	   # 1/0.15 mAb mg->nmol, MW = 150 kDa
  kainput1 <- 8;       # 1/d, mAb i.p.absorption rate (in suppl. kabs_mab)
  Vc       <- 0.003;	 # L, mAb volume of distribution (in suppl. Vd)
  kelmab   <- 0.1;     # 1/d, mAb elimination (In suppl. 0.15)
# PD-L1 monoclonal antibody
  d/dt(Ad1) <- -kainput1 * Ad1; # mg/d, mAb administration cmt (in suppl. mAb_ad/mAb_ip)
  d/dt(Ac1) <- kainput1 * Ad1 * convF1 - kelmab * Ac1; # nmol/d, mAb central cmt (in suppl. mAb_c)

  # AZD4635
  d/dt(Ac2) <- - (kel_ARinh * Ac2); # mg/d, A2AR inhibitor central cmt

  ##Tumor dynamics##
  Kd_ado    <- 1182; # nM, affinity of ado for A2AR (in suppl. Kd_A2AR_Ado, listed as 1.182 uM)
  Kd1       <- 30;	 # nM, mAb/PD-L1 binding affinity (in suppl. Kd)
  kel       <- 0.2;	 # 1/d, half-life of Tn naive
  kapo      <- 2.0;	 # 1/d, half-life of CTL
  kpro      <- 3;	   # 1/d, maximal T cells proliferation rate
  kdif      <- 3.2;	 # 1/d, maximal T cells differentiation rate
  d         <- 0.01; # 1/d, slow "spontaneous" tumor cells death rate (in suppl. d_0)
  Vmax_supr <- 0.7;  # the maximal adenosine effect on dTeff and APC suppression
                     # (In suppl. Fmax_A2Rs and value 1)

  # stops tumor growth if tumor volume smaller than 10 and time at least 7 days
  if (time > 7 & Tum < 10) {
    xf <- 0;
  }
  else{
    xf <- 1;
  }

  #------occupancy calculation------------
  Cc1 <- Ac1 / Vc;                  # nM, mAb
  Cc2 <- Ac2 / Vc_ARinh / MW * 1e6; # nM, A2AR inhibitor free
  PDL1free <- PDL1 / (1 + Cc1 / Kd1); # fraction unbound PDL1


  #-------variables------------

  A2ARoccup <- (Ado * sCf / Kd_ado) / (1 + (Ado * sCf / Kd_ado) + (Cc2 / Kd_ARinh)); # unitless, Ado occupied fraction of A2AR receptors
  Ado_suppr <- Vmax_supr * A2ARoccup / (A2ARoccup + IC50); # unitless, effect of ado-dependent A2AR occupancy
                                                           # on the activity of dTeff and APC (in suppl. A2ARs)
  TKR       <- (beff * CTL + d);                 # 1/d, tumor shrinkage rate (in suppl. TKR*Tum is TCD)
  Ag        <- TKR * Tum * (1 - Ado_suppr);      # uL/d, systemic antigen levels (in suppl. Ag_sys)
  PRfunc    <- (1 - PDL1free) * (1 - Ag / (Ag + sR)) * (1 - Ado_suppr); # unitless, Proliferation Rate/Immune activation rate (In suppl. IAR)
  TNinf     <- kLn * Ag / (Ag + sL);             # cells/day, CD8 precursor influx to tumor
  CD8_tot   <- CTL + TN;                         # cells, total CD8+ T cells
  ISC       <- (Ag / (Ag + sR));                 # unitless, immunosuppressive cells

  #------- Initial conditions -------
  Tum(0) <- TVin;     # uL, initial tumor volume

  #----------------- Model reactions-------
  Tum_gr       <- Tum * r * (1 - Tum / (TVmax)) * xf; # uL/d, tumor growth rate
  Tum_kill3    <- (beff * CTL + d) * Tum; # uL/d, tumor cell death (in suppl. TKR*Tum is TCD)
  CTL_dynamic1 <- TNinf + kpro * PRfunc * TN - kel * TN; # cells/d, influx of naive T cells
  CTL_dynamic2 <- kdif * PRfunc * TN; # cells/d, differentiation of naive T cells
  CTL_dynamic5 <- kapo * CTL; # cells/d, apoptosis of CTL

  d/dt(Tum)  <-  Tum_gr - Tum_kill3;          # uL/d, tumor volume (in suppl. TV or Tum)
  d/dt(TN)   <-  CTL_dynamic1 - CTL_dynamic2; # cells/d, naive T cells (In suppl. nTeff, Tpre)
  d/dt(CTL)  <-  CTL_dynamic2 - CTL_dynamic5; # cells/d, cytotoxic T-lymphocytes (In suppl. dTeff)
  d/dt(PDL1) <-  CTL / (CTL + Kp) - PDL1;	  #nM, PDL1
  d/dt(Ado)  <-  Vado * Tum / (Tum + Kado) - Ado; # nM/d, adenosine concentation
})