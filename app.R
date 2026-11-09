#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

# ============================================================
# ImForestSuite
# A Unified Forestry Analytics Dashboard
# Combines: ForAMat | DendroFit ML | DendroVol ML | TreeCarbonPro
#
# Developed by M. Iqbal Jeelani 
# SKUAST-Kashmir
# Contact: jeelani.miqbal@gmail.com
# YouTube: https://www.youtube.com/@Iqbalstat
# ============================================================

# ---- Libraries ----
library(shiny)
library(shinydashboard)
library(shinythemes)
library(fontawesome)
library(ggplot2)
library(DT)
library(plotly)
library(dplyr)
library(tidyverse)
library(minpack.lm)
library(Metrics)
library(caret)
library(shinyWidgets)
library(shinydashboardPlus)
library(randomForest)
library(e1071)
library(gbm)
library(rpart)
library(rpart.plot)
library(mgcv)
library(nls2)
library(neuralnet)
library(gridExtra)

# ---- Visual identity: Himalayan conifer palette ----
forest_pal <- c(deodar = "#1d4a3a", pine = "#2f6b52", lichen = "#8fb39a",
                larch = "#d9a21b", mist = "#6f8a86", ink = "#18241f")
forest_series <- c("#1d4a3a", "#d9a21b", "#4f8f74", "#6f8a86", "#a3612a",
                   "#8fb39a", "#3d5f7a", "#b8892b", "#7a4e6b", "#2f6b52")

theme_forest <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", colour = "#18241f", size = base_size + 1),
      axis.title = element_text(colour = "#4b5d55"),
      axis.text  = element_text(colour = "#6b7c74"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#e4ebe5", size = 0.4),
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.background  = element_rect(fill = "transparent", colour = NA),
      panel.background = element_rect(fill = "transparent", colour = NA),
      strip.text = element_text(face = "bold", colour = "#1d4a3a")
    )
}

# ============================================================
# UI
# ============================================================
ui <- dashboardPage(
  skin = "green",
  
  # ---- Header ----
  dashboardHeader(
    title = tags$span(class = "brand", tags$span(class = "brand-mark", icon("tree")), tags$span(class = "brand-name", "ImForestSuite")),
    titleWidth = 280,
    dropdownMenu(
      type = "notifications",
      badgeStatus = "success",
      icon = icon("info-circle"),
      notificationItem(text = "Developed by M.Iqbal Jeelani (SKUAST-Kashmir)", icon = icon("user")),
      notificationItem(text = "Contact: jeelani.miqbal@gmail.com", icon = icon("envelope")),
      notificationItem(text = "YouTube: @Iqbalstat", icon = icon("youtube"))
    )
  ),
  
  # ---- Sidebar ----
  dashboardSidebar(
    width = 280,
    sidebarMenu(
      id = "main_sidebar",
      menuItem("Home", tabName = "home", icon = icon("home")),
      menuItem("Forest Management & Mensuration", tabName = "ForAMat", icon = icon("chart-line"),
               menuSubItem("Forest Management", tabName = "management", icon = icon("seedling")),
               menuSubItem("Forest Mensuration", tabName = "mensuration", icon = icon("ruler")),
               menuSubItem("References", tabName = "references", icon = icon("book"))
      ),
      menuItem("DendroFit ML (H-D Modeling)", tabName = "DendroFitML", icon = icon("brain"),
               menuSubItem("Dashboard", tabName = "hd_dashboard", icon = icon("dashboard")),
               menuSubItem("Data", tabName = "hd_data", icon = icon("table")),
               menuSubItem("Traditional Models", tabName = "hd_models", icon = icon("chart-line")),
               menuSubItem("ML Models", tabName = "hd_ml_models", icon = icon("robot")),
               menuSubItem("Cross-Validation", tabName = "hd_validation", icon = icon("check-double"))
      ),
      menuItem("DendroVol ML (Volume Prediction)", tabName = "DendroVolML", icon = icon("chart-area")),
      menuItem("TreeCarbonPro (Carbon Stock)", tabName = "TreeCarbonPro", icon = icon("leaf"),
               menuSubItem("Input & Calculations", tabName = "carbon_input", icon = icon("calculator")),
               menuSubItem("Results", tabName = "carbon_results", icon = icon("chart-bar"))
      ),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    ),
    
    # Conditional: DendroFitML sidebar controls
    conditionalPanel(
      condition = "input.main_sidebar == 'hd_data' || input.main_sidebar == 'hd_dashboard'",
      hr(),
      fileInput("hd_dataFile", "Upload CSV (H-D Data)", accept = ".csv"),
      actionButton("hd_demo", "Or load 150 demo trees", icon = icon("seedling"), class = "btn-demo"),
      selectInput("hd_diameterCol", "Diameter Column:", ""),
      selectInput("hd_heightCol", "Height Column:", "")
    ),
    conditionalPanel(
      condition = "input.main_sidebar == 'hd_models' || input.main_sidebar == 'hd_validation'",
      hr(),
      checkboxGroupInput("hd_models", "Traditional H-D Models:",
                         choices = c("Naslund 1" = "naslund1", "Curtis" = "curtis",
                                     "Michailoff" = "michailoff", "Naslund 2" = "naslund2",
                                     "Power" = "power", "Naslund 3" = "naslund3",
                                     "Naslund 4" = "naslund4", "Michaelis-Menten" = "michail_menten",
                                     "Michaelis-Menten 2" = "michail_menten2", "Wykoff" = "wykoff"),
                         selected = c("naslund1", "curtis", "michailoff", "power"))
    ),
    conditionalPanel(
      condition = "input.main_sidebar == 'hd_ml_models' || input.main_sidebar == 'hd_validation'",
      hr(),
      checkboxGroupInput("hd_ml_models", "ML Models (H-D):",
                         choices = c("Decision Tree" = "dt", "Random Forest" = "rf",
                                     "SVM" = "svm", "Gradient Boosting" = "gbm",
                                     "Neural Network (ANN)" = "ann"),
                         selected = c("dt", "rf")),
      checkboxInput("hd_doCrossValidation", "Perform Cross-Validation", TRUE),
      conditionalPanel(
        condition = "input.hd_doCrossValidation == true",
        sliderInput("hd_trainPercent", "Training %:", min = 50, max = 90, value = 80)
      ),
      numericInput("hd_ntree", "RF - No. of Trees:", 500, min = 10, max = 2000),
      numericInput("hd_rf_mtry", "RF - mtry:", 1, min = 1, max = 5),
      numericInput("hd_n_trees_gbm", "GBM - Trees:", 100, min = 10, max = 500),
      selectInput("hd_svm_kernel", "SVM Kernel:", choices = c("Radial" = "radial", "Linear" = "linear", "Polynomial" = "polynomial")),
      numericInput("hd_svm_cost", "SVM Cost:", 1, min = 0.1, max = 100, step = 0.1),
      sliderInput("hd_cp", "DT Complexity:", min = 0.001, max = 0.1, value = 0.01, step = 0.001),
      numericInput("hd_ann_hidden1", "ANN - Hidden Neurons (Layer 1):", 8, min = 2, max = 30),
      numericInput("hd_ann_hidden2", "ANN - Hidden Neurons (Layer 2):", 4, min = 0, max = 20),
      numericInput("hd_ann_thresh",  "ANN - Threshold:", 0.05, min = 0.01, max = 0.5, step = 0.01)
    ),
    conditionalPanel(
      condition = "input.main_sidebar == 'hd_ml_models' || input.main_sidebar == 'hd_validation' || input.main_sidebar == 'hd_models'",
      div(style = "padding: 10px;",
          actionButton("hd_runAnalysis", "Run H-D analysis", icon = icon("play"), class = "btn-larch btn-block")
      )
    ),
    
    # Conditional: TreeCarbonPro sidebar inputs
    conditionalPanel(
      condition = "input.main_sidebar == 'carbon_input' || input.main_sidebar == 'carbon_results'",
      hr(),
      numericInput("tc_dbh", "DBH (cm):", value = 30, min = 1),
      numericInput("tc_height", "Height (m):", value = 15, min = 1),
      numericInput("tc_density", "Wood Density (g/cm³):", value = 0.6, min = 0.1),
      numericInput("tc_volume", "Stem Volume (m³):", value = 0.5, min = 0.01),
      numericInput("tc_bef", "BEF:", value = 1.3, min = 0.5),
      numericInput("tc_rs_ratio", "Root-to-Shoot Ratio:", value = 0.24, min = 0.01),
      numericInput("tc_carbon_frac", "Carbon Fraction:", value = 0.47, min = 0.1, max = 0.6),
      div(style = "padding: 10px;",
          actionButton("tc_calc", "Estimate carbon stock", icon = icon("leaf"), class = "btn-larch btn-block")
      )
    ),
    
    # Footer
    tags$div(class = "sidebar-footer",
             tags$div(class = "sf-name", "M. Iqbal Jeelani"),
             tags$div(class = "sf-org", "SKUAST-Kashmir"),
             tags$div(class = "sf-links",
                      a(icon("envelope"), href = "mailto:jeelani.miqbal@gmail.com", title = "Email"),
                      a(icon("youtube"), href = "https://www.youtube.com/@Iqbalstat", target = "_blank", title = "YouTube @Iqbalstat")
             )
    )
  ),
  
  # ---- Body ----
  dashboardBody(
    tags$head(
      tags$link(rel = "preconnect", href = "https://fonts.googleapis.com"),
      tags$link(rel = "stylesheet",
                href = "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap"),
      tags$script(HTML("$(function () {
  var reduce = window.matchMedia && window.matchMedia(\"(prefers-reduced-motion: reduce)\").matches;
  function debounce(fn, ms) { var t; return function (e) { clearTimeout(t); t = setTimeout(function () { fn(e); }, ms); }; }
  function fmt(x, d) { return x.toLocaleString(\"en-IN\", { minimumFractionDigits: d, maximumFractionDigits: d }); }

  /* ---------- Hero: try-a-tree explorer ---------- */
  var NS = \"http://www.w3.org/2000/svg\";
  function drawTree(D, H) {
    var g = 244, s = 210 / 45, h = H * s;
    var tw = Math.min(22, 3 + D * 0.17), cx = 128;
    var top = g - h, cb = g - h * 0.2, ch = cb - top;
    var maxW = Math.min(190, 18 + h * 0.4 + D * 0.5);
    var trunk = document.getElementById(\"tx-trunk\");
    trunk.setAttribute(\"d\", \"M\" + (cx - tw / 2) + \",\" + g + \" L\" + (cx - tw * 0.18) + \",\" + top + \" L\" + (cx + tw * 0.18) + \",\" + top + \" L\" + (cx + tw / 2) + \",\" + g + \" Z\");
    var tiers = Math.max(3, Math.min(8, 3 + Math.round(h / 40))), d1 = \"\", d2 = \"\";
    for (var i = 0; i < tiers; i++) {
      var tt = top + i * (ch / tiers) * 0.82, tb = Math.min(cb, tt + (ch / tiers) * 1.45);
      var w = maxW * (0.3 + 0.7 * (i + 1) / tiers);
      var p = \"M\" + cx + \",\" + (tt - 2) + \" L\" + (cx - w / 2) + \",\" + tb + \" Q\" + cx + \",\" + (tb - 6) + \" \" + (cx + w / 2) + \",\" + tb + \" Z \";
      if (i % 2) d2 += p; else d1 += p;
    }
    document.getElementById(\"tx-crown-a\").setAttribute(\"d\", d1);
    document.getElementById(\"tx-crown-b\").setAttribute(\"d\", d2);
    var m = document.getElementById(\"tx-mark\");
    m.setAttribute(\"y1\", top); m.setAttribute(\"y2\", g);
    var ml = document.getElementById(\"tx-mark-lbl\");
    ml.setAttribute(\"y\", top + 4); ml.textContent = fmt(H, 0) + \" m\";
    /* stump disk with growth rings */
    var disk = document.getElementById(\"tx-disk\"); while (disk.firstChild) disk.removeChild(disk.firstChild);
    var r = 6 + Math.min(34, D * 0.34), n = Math.max(3, Math.round(r / 3.2));
    for (var k = n; k >= 1; k--) {
      var c = document.createElementNS(NS, \"circle\");
      c.setAttribute(\"cx\", 42); c.setAttribute(\"cy\", 42); c.setAttribute(\"r\", r * k / n);
      c.setAttribute(\"class\", k === n ? \"bark\" : (k % 2 ? \"ring-a\" : \"ring-b\"));
      disk.appendChild(c);
    }
  }
  function updateTree() {
    var D = +$(\"#tx_dbh\").val(), H = +$(\"#tx_h\").val();
    $(\"#tx_dbh_v\").text(D + \" cm\"); $(\"#tx_h_v\").text(H + \" m\");
    var ba = 0.7854 * Math.pow(D / 100, 2), v = ba * H * 0.5;
    var agb = 0.0509 * 0.6 * D * D * H, c = agb * 1.24 * 0.47;
    $(\"#tx_ba\").text(fmt(ba, 3)); $(\"#tx_vol\").text(fmt(v, 2));
    $(\"#tx_agb\").text(fmt(agb, 0)); $(\"#tx_c\").text(fmt(c, 0));
    $(\"#tx_co2\").text(fmt(c * 44 / 12, 0));
    drawTree(D, H);
  }
  if ($(\"#tx_dbh\").length) { $(document).on(\"input\", \"#tx_dbh, #tx_h\", updateTree); updateTree(); }

  /* ---------- Calculators recalculate as you type ---------- */
  $(document).on(\"input change\", \".formula-card input\", debounce(function (e) {
    $(e.target).closest(\".formula-card\").find(\"button.calculate-btn\").first().click();
  }, 350));
  var filled = {};
  function fillTab(name) {
    if (filled[name] || [\"management\", \"mensuration\"].indexOf(name) < 0) return;
    filled[name] = true;
    setTimeout(function () { $(\"#shiny-tab-\" + name + \" .formula-card button.calculate-btn\").each(function () { $(this).click(); }); }, 250);
  }
  $(document).on(\"shown.bs.tab\", \"a[data-toggle=tab]\", function (e) { fillTab($(e.target).attr(\"data-value\")); });

  /* ---------- Carbon results follow the sidebar inputs ---------- */
  $(document).on(\"input change\", \"#tc_dbh, #tc_height, #tc_density, #tc_volume, #tc_bef, #tc_rs_ratio, #tc_carbon_frac\", debounce(function () {
    if ($(\"#shiny-tab-carbon_results\").hasClass(\"active\")) $(\"#tc_calc\").click();
  }, 400));

  /* ---------- Count-up on value boxes ---------- */
  $(document).on(\"shiny:value\", function (e) {
    if (!/^(tc_vb_|hd_totalObs)/.test(e.name) || reduce) return;
    setTimeout(function () {
      var h = $(\"#\" + e.name + \" h3\").get(0); if (!h || !h.firstChild || h.firstChild.nodeType !== 3) return;
      var node = h.firstChild, raw = node.nodeValue.replace(/,/g, \"\"), to = parseFloat(raw);
      if (isNaN(to)) return;
      var dec = (raw.split(\".\")[1] || \"\").length, t0 = null;
      function step(ts) { if (!t0) t0 = ts; var p = Math.min(1, (ts - t0) / 700), q = 1 - Math.pow(1 - p, 3);
        node.nodeValue = fmt(to * q, dec); if (p < 1) requestAnimationFrame(step); }
      requestAnimationFrame(step);
    }, 0);
  });
});
")),
      tags$style(HTML("        :root {
          --deodar: #1d4a3a; --deodar-deep: #12332a; --pine: #2f6b52; --lichen: #8fb39a;
          --larch: #d9a21b; --larch-deep: #b9860f; --mist: #6f8a86; --ink: #18241f;
          --bark: #eef3ef; --line: #dbe5de; --card: #ffffff;
        }
        body, .content-wrapper, .main-sidebar, .box, .form-control, .btn, table { font-family: 'Figtree', system-ui, sans-serif; }
        h1, h2, h3, h4, .brand-name, .box-title, .small-box h3, .info-box-number { font-family: 'Bricolage Grotesque', 'Figtree', sans-serif; letter-spacing: -0.01em; }
        body { color: var(--ink); }
        .content-wrapper, .right-side { background: var(--bark); }
        .content { padding: 22px 24px 40px; }

        /* ---- Header ---- */
        .skin-green .main-header .logo, .skin-green .main-header .logo:hover { background: var(--deodar-deep); }
        .skin-green .main-header .navbar { background: var(--deodar); }
        .skin-green .main-header .navbar .sidebar-toggle:hover { background: rgba(255,255,255,0.08); }
        .brand { display: inline-flex; align-items: center; gap: 10px; }
        .brand-mark { width: 30px; height: 30px; border-radius: 50%; background: var(--larch); color: var(--deodar-deep);
                      display: inline-flex; align-items: center; justify-content: center; font-size: 14px;
                      box-shadow: 0 0 0 3px rgba(217,162,27,0.25); }
        .brand-name { font-weight: 800; font-size: 19px; color: #fff; }

        /* ---- Sidebar ---- */
        .skin-green .main-sidebar, .skin-green .left-side { background: var(--deodar-deep); }
        .skin-green .sidebar-menu > li > a { color: #c9dbd0; border-left: 3px solid transparent; padding: 12px 15px; font-weight: 500; font-size: 13.5px; }
        .skin-green .sidebar-menu > li:hover > a { background: rgba(255,255,255,0.05); color: #fff; }
        .skin-green .sidebar-menu > li.active > a, .skin-green .sidebar-menu > li.menu-open > a {
          background: rgba(255,255,255,0.07); color: #fff; border-left-color: var(--larch); }
        .skin-green .sidebar-menu > li > .treeview-menu { background: rgba(0,0,0,0.18); }
        .skin-green .treeview-menu > li > a { color: #a9c1b3; }
        .skin-green .treeview-menu > li.active > a, .skin-green .treeview-menu > li > a:hover { color: var(--larch); }
        .main-sidebar .control-label, .main-sidebar label { color: #dfe9e2; font-weight: 500; }
        .main-sidebar hr { border-top: 1px solid rgba(255,255,255,0.1); margin: 14px 15px; }
        .main-sidebar .form-control { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.14); color: #fff; }
        .main-sidebar .selectize-input { background: rgba(255,255,255,0.95); }
        .main-sidebar .shiny-input-container { padding-left: 15px; padding-right: 15px; }
        .main-sidebar .irs-bar, .main-sidebar .irs-single { background: var(--larch); border-color: var(--larch); }
        .sidebar-footer { margin: 28px 15px 20px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.1); color: #9fb7a9; font-size: 12px; }
        .sf-name { color: #fff; font-weight: 600; font-size: 13px; }
        .sf-links { margin-top: 10px; display: flex; gap: 8px; }
        .sf-links a { width: 32px; height: 32px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
                      background: rgba(255,255,255,0.08); color: #dfe9e2; }
        .sf-links a:hover, .sf-links a:focus { background: var(--larch); color: var(--deodar-deep); }

        /* ---- Page headings ---- */
        .page-head { margin: 4px 0 22px; max-width: 760px; }
        .page-head h2 { font-size: 30px; font-weight: 800; color: var(--deodar); margin: 0 0 6px; }
        .page-head p { color: #56685f; font-size: 15px; margin: 0; line-height: 1.55; }

        /* ---- Boxes ---- */
        .box { border-radius: 12px; border: 1px solid var(--line); border-top: 1px solid var(--line); box-shadow: 0 1px 2px rgba(18,51,42,0.06); }
        .box.box-solid > .box-header { border-radius: 11px 11px 0 0; padding: 13px 16px; }
        .box.box-solid.box-success, .box.box-solid.box-primary, .box.box-solid.box-info { border: 1px solid var(--line); }
        .box.box-solid.box-success > .box-header { background: var(--deodar); color: #fff; }
        .box.box-solid.box-primary > .box-header { background: var(--pine); color: #fff; }
        .box.box-solid.box-info > .box-header { background: #3d5f7a; color: #fff; }
        .box-header .box-title { font-size: 16px; font-weight: 700; }
        .box-body { padding: 16px 18px; }
        .nav-tabs-custom { border-radius: 12px; border: 1px solid var(--line); box-shadow: none; overflow: hidden; }
        .nav-tabs-custom > .nav-tabs > li.active { border-top-color: var(--larch); }
        .nav-tabs-custom > .nav-tabs > li.active > a { color: var(--deodar); font-weight: 600; }
        .nav-tabs > li > a { color: #56685f; }
        .nav-tabs > li.active > a, .nav-tabs > li.active > a:focus { color: var(--deodar); font-weight: 600; border-top: 2px solid var(--larch); }

        /* ---- Formula cards (management & mensuration) ---- */
        .formula-card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 20px 22px; margin-bottom: 22px; }
        .formula-card h3 { font-size: 20px; font-weight: 700; color: var(--deodar); border: none; padding: 0; margin: 0 0 4px; }
        .description { color: #56685f; margin-bottom: 8px; }
        .math-formula { background: #f6f9f6; border: none; border-left: 3px solid var(--larch); border-radius: 0 8px 8px 0; padding: 8px 14px; margin-top: 10px; color: #3e5047; }
        .input-section { background: transparent; padding: 14px 0 0; margin-top: 6px; border-top: 1px dashed var(--line); }
        .input-section h4 { font-size: 14px; font-weight: 600; color: var(--mist); margin-top: 0; }
        .result-display { background: transparent; border: none; padding: 6px 0 0; margin-top: 4px; }
        .result-display pre { background: var(--deodar); color: #fff; border: none; border-radius: 8px; font-family: 'Figtree', sans-serif;
                              font-size: 16px; font-weight: 600; padding: 10px 14px; white-space: pre-wrap; }

        /* ---- Buttons ---- */
        .btn { border-radius: 8px; font-weight: 600; transition: background-color .15s ease, box-shadow .15s ease; }
        .calculate-btn, .btn-forest { background: var(--deodar); color: #fff !important; border: 1px solid var(--deodar); padding: 7px 16px; }
        .calculate-btn:hover, .btn-forest:hover, .calculate-btn:focus, .btn-forest:focus { background: var(--pine); border-color: var(--pine); }
        .btn-larch { background: var(--larch); color: var(--deodar-deep) !important; border: 1px solid var(--larch); padding: 9px 14px; font-weight: 700; }
        .btn-larch:hover, .btn-larch:focus { background: #e6b23a; border-color: #e6b23a; }
        .main-sidebar .btn-larch { width: calc(100% - 30px); margin: 10px 15px; }
        .btn-ghost { background: transparent; color: var(--deodar) !important; border: 1px solid var(--deodar); }
        .btn-ghost:hover { background: rgba(29,74,58,0.06); }
        .btn:focus-visible, a:focus-visible, .form-control:focus { outline: 3px solid rgba(217,162,27,0.55); outline-offset: 2px; }

        /* ---- Inputs ---- */
        .form-control, .selectize-input { border-radius: 8px; border-color: #cfdcd3; box-shadow: none; }
        .form-control:focus { border-color: var(--pine); box-shadow: none; }
        .irs--shiny .irs-bar, .irs--shiny .irs-single, .irs--shiny .irs-from, .irs--shiny .irs-to { background: var(--pine); border-color: var(--pine); }
        .well { background: var(--card); border: 1px solid var(--line); border-radius: 12px; box-shadow: none; }
        .well h4 { font-size: 15px; font-weight: 700; color: var(--deodar); }

        /* ---- Value boxes ---- */
        .small-box { border-radius: 12px; overflow: hidden; box-shadow: none; }
        .small-box h3 { font-size: 30px; font-weight: 800; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .small-box h3 small { font-size: 15px; color: inherit; opacity: .8; }
        .small-box .icon-large, .small-box .icon { opacity: .18; }
        .bg-green { background: var(--deodar) !important; }
        .bg-olive { background: var(--pine) !important; }
        .bg-teal { background: #3d5f7a !important; }
        .bg-blue { background: var(--pine) !important; }
        .bg-purple { background: #3d5f7a !important; }
        .bg-yellow, .bg-orange { background: var(--larch) !important; color: var(--deodar-deep) !important; }
        .small-box.bg-yellow .small-box-footer, .small-box.bg-orange .small-box-footer { color: var(--deodar-deep); }

        /* ---- Tables & text output ---- */
        table.dataTable thead th { background: #f3f7f4; color: var(--deodar); font-weight: 700; border-bottom: 2px solid var(--lichen) !important; }
        table.dataTable tbody tr:hover { background: #f6faf7 !important; }
        .dataTables_wrapper .dataTables_paginate .paginate_button.current { background: var(--deodar) !important; color: #fff !important; border: none !important; border-radius: 6px; }
        pre { background: #f6f9f6; border: 1px solid var(--line); border-radius: 8px; color: #24332c; }

        /* ---- Carbon equations ---- */
        .equation-box { background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--larch); border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 12px; }
        .ref-title { font-weight: 700; color: var(--deodar); margin-bottom: 4px; }
        .math { font-style: italic; color: #3e5047; margin: 0 0 2px; }
        .hint { color: var(--mist); font-weight: 600; margin-top: 8px; }
        .ref-list li { color: #56685f; line-height: 1.7; }

        /* ---- Home: hero ---- */
        .hero { position: relative; overflow: hidden; border-radius: 16px; background: linear-gradient(180deg, #e3ece6 0%, #cfdfd4 52%, #b9cebf 100%);
                min-height: 360px; padding: 44px 48px 150px; margin-bottom: 26px; }
        .hero-sun { position: absolute; right: 9%; top: 34px; width: 150px; height: 150px; border-radius: 50%;
                    background: repeating-radial-gradient(circle, rgba(217,162,27,0.95) 0 6px, rgba(217,162,27,0.55) 6px 9px, rgba(217,162,27,0.9) 9px 15px, rgba(217,162,27,0.45) 15px 17px); opacity: .9; }
        .hero-copy { position: relative; z-index: 2; max-width: 600px; }
        .hero h1 { font-size: 54px; line-height: 1.02; font-weight: 800; color: var(--deodar-deep); margin: 0 0 14px; letter-spacing: -0.03em; }
        .hero p.lede { font-size: 17px; line-height: 1.55; color: #33463d; margin: 0 0 18px; max-width: 520px; }
        .hero-by { font-size: 13px; color: #4b5d55; }
        .hero svg.ridges { position: absolute; left: 0; bottom: 0; width: 100%; height: 230px; z-index: 1; }
        .hero svg .r1 { fill: #8fae9b; } .hero svg .r2 { fill: #2f6b52; } .hero svg .r3 { fill: #12332a; }
        .hero svg .r1, .hero svg .r2, .hero svg .r3 { transform-box: fill-box; transform-origin: bottom; animation: rise 1.1s cubic-bezier(.2,.7,.2,1) both; }
        .hero svg .r2 { animation-delay: .12s; } .hero svg .r3 { animation-delay: .24s; }
        @keyframes rise { from { transform: translateY(40px); opacity: 0; } to { transform: none; opacity: 1; } }
        .hero-facts { position: absolute; z-index: 3; text-shadow: 0 1px 3px rgba(18,51,42,0.6); left: 48px; right: 48px; bottom: 22px; display: flex; gap: 36px; flex-wrap: wrap; color: #e5efe8; }
        .hero-facts div { font-size: 13px; }
        .hero-facts b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 26px; color: #fff; line-height: 1.1; }

        /* ---- Home: modules ---- */
        .modules { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px; margin-bottom: 26px; }
        .module { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 22px 20px 18px; display: flex; flex-direction: column; }
        .module-icon { width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 19px;
                       background: #e6efe9; color: var(--deodar); margin-bottom: 14px; }
        .module h3 { font-size: 20px; font-weight: 700; color: var(--deodar-deep); margin: 0 0 8px; }
        .module p { color: #56685f; line-height: 1.55; flex: 1; }
        .module .tags { margin: 6px 0 14px; display: flex; flex-wrap: wrap; gap: 6px; }
        .module .tags span { background: #f2f6f3; border: 1px solid var(--line); color: #3e5047; border-radius: 999px; padding: 2px 10px; font-size: 12px; }
        .module .btn { align-self: flex-start; }
        .module.carbon .module-icon { background: #fbf1d8; color: var(--larch-deep); }

        /* ---- Home: how it works ---- */
        .howto { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 22px 24px; }
        .howto h3 { margin: 0 0 14px; font-size: 20px; font-weight: 700; color: var(--deodar-deep); }
        .howto-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 22px; }
        .howto-steps .n { width: 28px; height: 28px; border-radius: 50%; background: var(--larch); color: var(--deodar-deep); font-weight: 800;
                          display: inline-flex; align-items: center; justify-content: center; margin-bottom: 8px; }
        .howto-steps h4 { font-size: 15px; font-weight: 700; margin: 0 0 4px; color: var(--ink); font-family: 'Figtree', sans-serif; }
        .howto-steps p { color: #56685f; margin: 0; line-height: 1.5; }

        /* ---- About ---- */
        .about-lead h3 { font-size: 28px; font-weight: 800; color: var(--deodar); margin-top: 4px; }
        .about-lead p { font-size: 16px; color: #56685f; max-width: 640px; }
        .about-section p, .about-section li { line-height: 1.7; }

        /* ---- Busy indicator ---- */
        html.shiny-busy .main-header::after { content: ''; position: fixed; top: 0; left: 0; height: 3px; width: 35%; z-index: 2000;
                                              background: var(--larch); animation: busy 1.1s ease-in-out infinite; }
        @keyframes busy { from { transform: translateX(-100%); } to { transform: translateX(300%); } }
        .shiny-notification { background: #fff; color: var(--ink); opacity: 1; box-shadow: 0 6px 20px rgba(18,51,42,0.15); border-radius: 10px; border-left: 4px solid var(--larch); font-family: 'Figtree', sans-serif; }

        /* ---- Hero v2: two columns with the tree explorer ---- */
        .hero { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr); gap: 36px; align-items: start; padding-bottom: 120px; min-height: 520px; }
        .hero-copy { padding-top: 14px; }
        .hero-sun { display: none; }
        .tx-card { position: relative; z-index: 2; background: rgba(255,255,255,0.86); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
                   border: 1px solid rgba(255,255,255,0.9); border-radius: 16px; padding: 18px 20px 16px; box-shadow: 0 18px 40px -18px rgba(18,51,42,0.45); }
        .tx-card h4 { margin: 0 0 2px; font-size: 18px; font-weight: 800; color: var(--deodar-deep); font-family: 'Bricolage Grotesque', sans-serif; }
        .tx-card .tx-sub { font-size: 12.5px; color: var(--mist); margin-bottom: 8px; }
        .tx-body { display: grid; grid-template-columns: 200px minmax(0, 1fr); gap: 16px; align-items: end; }
        .tx-svg { width: 100%; height: auto; display: block; }
        .tx-svg .sky { fill: #eef4f0; }
        .tx-svg .ground { fill: #c9d9cd; }
        .tx-svg #tx-trunk { fill: #6b4a2e; }
        .tx-svg #tx-crown-a { fill: var(--deodar); }
        .tx-svg #tx-crown-b { fill: var(--pine); }
        .tx-svg #tx-mark { stroke: var(--larch-deep); stroke-width: 1.2; stroke-dasharray: 3 3; }
        .tx-svg #tx-mark-lbl { fill: var(--larch-deep); font: 700 11px Figtree, sans-serif; }
        .tx-svg .person { fill: #3e5047; }
        .tx-svg .person-lbl { fill: #6b7c74; font: 9px Figtree, sans-serif; }
        .tx-ctrl { margin-bottom: 10px; }
        .tx-ctrl label { display: flex; justify-content: space-between; font-weight: 600; font-size: 13px; color: var(--ink); margin: 0 0 4px; }
        .tx-ctrl label output { color: var(--deodar); font-variant-numeric: tabular-nums; }
        .tx-ctrl input[type=range] { width: 100%; accent-color: var(--larch-deep); cursor: pointer; }
        .tx-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; margin-top: 6px; }
        .tx-stats div { font-size: 12px; color: #56685f; line-height: 1.25; }
        .tx-stats b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 22px; color: var(--deodar-deep); font-variant-numeric: tabular-nums; }
        .tx-stats .c b { color: var(--larch-deep); }
        .tx-disk-wrap { display: flex; align-items: center; gap: 10px; margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--line); font-size: 12px; color: #56685f; }
        .tx-disk { width: 84px; height: 84px; flex: none; }
        .tx-disk .bark { fill: #6b4a2e; } .tx-disk .ring-a { fill: #e9c98f; } .tx-disk .ring-b { fill: #d9b06a; }
        .tx-note { font-size: 11.5px; color: var(--mist); margin-top: 8px; }

        /* ---- Live badges and uploads ---- */
        .live-pill { display: inline-flex; align-items: center; gap: 6px; background: #fff; border: 1px solid var(--line); border-radius: 999px;
                     padding: 3px 10px 3px 8px; font-size: 12.5px; font-weight: 600; color: var(--deodar); margin-top: 10px; }
        .live-pill::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: #3fa36b; box-shadow: 0 0 0 0 rgba(63,163,107,0.6); animation: pulse 2s infinite; }
        @keyframes pulse { 70% { box-shadow: 0 0 0 7px rgba(63,163,107,0); } 100% { box-shadow: 0 0 0 0 rgba(63,163,107,0); } }
        .btn-file { background: var(--larch) !important; color: var(--deodar-deep) !important; border: none !important; font-weight: 700; }
        .shiny-input-container .input-group .form-control[readonly] { background: rgba(255,255,255,0.92); color: var(--ink); }
        .progress-bar { background-color: var(--larch); }
        .main-sidebar .btn-demo { width: calc(100% - 30px); margin: -4px 15px 6px; background: transparent; color: #dfe9e2 !important; border: 1px dashed rgba(255,255,255,0.35); }
        .main-sidebar .btn-demo:hover, .main-sidebar .btn-demo:focus { border-color: var(--larch); color: var(--larch) !important; background: rgba(217,162,27,0.08); }
        .result-display pre { transition: background-color .3s ease; }
        .formula-card:focus-within { border-color: var(--lichen); box-shadow: 0 0 0 3px rgba(143,179,154,0.25); }

        .pagination > .active > a, .pagination > .active > a:hover, .pagination > .active > a:focus { background: var(--deodar); border-color: var(--deodar); color: #fff; }
        .dataTables_wrapper .dataTables_paginate .paginate_button.current a { color: #fff !important; }
        .skin-green .sidebar-menu > li > a { padding-right: 32px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        @media (max-width: 1100px) { .hero { grid-template-columns: 1fr; } }
        @media (max-width: 767px) { .tx-body { grid-template-columns: 1fr; } .tx-svg { max-width: 220px; margin: 0 auto; } }
        @media (prefers-reduced-motion: reduce) { .live-pill::before { animation: none; } }

        @media (max-width: 1200px) { .modules { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 767px) {
          .modules, .howto-steps { grid-template-columns: 1fr; }
          .hero { padding: 28px 22px 170px; } .hero h1 { font-size: 38px; } .hero-sun { width: 90px; height: 90px; right: 18px; top: 18px; opacity: .5; }
          .hero-facts { left: 22px; right: 22px; gap: 18px; } .hero-facts b { font-size: 20px; }
          .content { padding: 14px 12px 30px; }
        }
        @media (prefers-reduced-motion: reduce) { .hero svg .r1, .hero svg .r2, .hero svg .r3, html.shiny-busy .main-header::after { animation: none; } }
"))
    ),
    withMathJax(),
    
    tabItems(
      # ============================================================
      # HOME TAB
      # ============================================================
      tabItem(tabName = "home",
              div(class = "hero",
                  div(class = "hero-copy",
                      h1("Measure the forest, tree by tree."),
                      p(class = "lede", "Mensuration formulas, height-diameter and volume models, and carbon estimates in one workspace. Upload your field data and compare classical allometry with machine learning side by side."),
                      div(class = "hero-by", "M. Iqbal Jeelani, SKUAST-Kashmir")
                  ),
                  div(class = "tx-card",
                      h4("Try a tree"),
                      div(class = "tx-sub", "Drag the sliders to grow a deodar and watch the numbers follow."),
                      div(class = "tx-body",
                          HTML('<svg class="tx-svg" viewBox="0 0 220 256" role="img" aria-label="Tree drawn to scale">
                                <rect class="sky" x="0" y="0" width="220" height="244" rx="10"/>
                                <rect class="ground" x="0" y="242" width="220" height="14" rx="4"/>
                                <path id="tx-trunk"/><path id="tx-crown-b"/><path id="tx-crown-a"/>
                                <line id="tx-mark" x1="206" x2="206" y1="40" y2="244"/>
                                <text id="tx-mark-lbl" x="202" y="40" text-anchor="end"></text>
                                <g class="person"><circle cx="30" cy="235.3" r="1.5"/><rect x="29" y="236.8" width="2" height="6.2" rx="1"/></g>
                                <text class="person-lbl" x="36" y="241">1.7 m</text></svg>'),
                          div(
                            div(class = "tx-ctrl",
                                tags$label(`for` = "tx_dbh", "Diameter (DBH)", tags$output(id = "tx_dbh_v")),
                                tags$input(type = "range", id = "tx_dbh", min = 5, max = 100, value = 32, step = 1)),
                            div(class = "tx-ctrl",
                                tags$label(`for` = "tx_h", "Height", tags$output(id = "tx_h_v")),
                                tags$input(type = "range", id = "tx_h", min = 3, max = 45, value = 22, step = 1)),
                            div(class = "tx-stats",
                                div(tags$b(id = "tx_ba", "-"), HTML("Basal area, m&sup2;")),
                                div(tags$b(id = "tx_vol", "-"), HTML("Stem volume, m&sup3;")),
                                div(tags$b(id = "tx_agb", "-"), "Above-ground biomass, kg"),
                                div(class = "c", tags$b(id = "tx_c", "-"), "Carbon incl. roots, kg")
                            )
                          )
                      ),
                      div(class = "tx-disk-wrap",
                          HTML('<svg class="tx-disk" viewBox="0 0 84 84" aria-hidden="true"><g id="tx-disk"></g></svg>'),
                          div(HTML("Stump cross-section at breast height. This tree has locked away about <b id='tx_co2'>-</b> kg of CO&#8322;."))
                      ),
                      div(class = "tx-note", HTML("Assumes wood density 0.6 g/cm&sup3;, form factor 0.5, root:shoot 0.24 and carbon fraction 0.47 (Chave 2005, IPCC)."))
                  ),
                  HTML('<svg class="ridges" viewBox="0 0 1200 260" preserveAspectRatio="none" aria-hidden="true">
                        <path class="r1" d="M0,260 L0,134 L22,142 L33,123 L44,142 L44,149 L55,131 L66,149 L66,154 L88,158 L99,140 L110,158 L110,159 L121,142 L132,159 L132,157 L143,146 L154,157 L154,154 L176,149 L198,143 L209,130 L220,143 L220,137 L242,131 L264,126 L286,122 L308,120 L330,119 L341,98 L352,119 L352,120 L363,108 L374,120 L374,122 L385,107 L396,122 L396,125 L418,127 L429,109 L440,127 L440,129 L462,129 L484,128 L506,125 L517,114 L528,125 L528,121 L539,101 L550,121 L550,115 L572,108 L594,100 L616,93 L638,87 L649,66 L660,87 L660,83 L682,80 L693,62 L704,80 L704,80 L726,83 L748,87 L770,93 L781,70 L792,93 L792,100 L803,84 L814,100 L814,107 L836,114 L847,97 L858,114 L858,120 L869,101 L880,120 L880,125 L902,127 L924,128 L946,128 L968,126 L990,124 L1012,121 L1034,119 L1056,119 L1078,119 L1100,122 L1122,126 L1144,131 L1155,111 L1166,131 L1166,137 L1188,143 L1200,260 Z"/>
                        <path class="r2" d="M0,260 L0,176 L16,180 L24,157 L32,180 L32,184 L40,168 L48,184 L48,188 L56,169 L64,188 L64,191 L72,174 L80,191 L80,193 L96,195 L104,174 L112,195 L112,195 L120,164 L128,195 L128,195 L136,171 L144,195 L144,194 L152,162 L160,194 L160,193 L176,190 L192,188 L200,165 L208,188 L208,185 L216,153 L224,185 L224,182 L240,180 L248,161 L256,180 L256,177 L264,157 L272,177 L272,175 L280,149 L288,175 L288,174 L296,158 L304,174 L304,173 L312,151 L320,173 L320,173 L328,140 L336,173 L336,173 L344,148 L352,173 L352,174 L360,146 L368,174 L368,175 L376,143 L384,175 L384,176 L400,178 L416,179 L432,180 L440,157 L448,180 L448,181 L456,154 L464,181 L464,181 L472,164 L480,181 L480,180 L488,162 L496,180 L496,179 L504,162 L512,179 L512,177 L520,159 L528,177 L528,174 L536,152 L544,174 L544,171 L552,140 L560,171 L560,168 L568,149 L576,168 L576,164 L584,142 L592,164 L592,160 L600,142 L608,160 L608,156 L624,152 L640,149 L648,124 L656,149 L656,146 L664,128 L672,146 L672,144 L680,124 L688,144 L688,143 L704,143 L712,127 L720,143 L720,143 L736,145 L744,126 L752,145 L752,147 L760,131 L768,147 L768,149 L776,116 L784,149 L784,152 L800,156 L808,135 L816,156 L816,159 L824,140 L832,159 L832,162 L848,165 L856,135 L864,165 L864,168 L872,148 L880,168 L880,170 L896,171 L912,172 L928,172 L944,171 L960,170 L976,169 L984,144 L992,169 L992,168 L1000,152 L1008,168 L1008,167 L1016,146 L1024,167 L1024,166 L1032,138 L1040,166 L1040,165 L1056,165 L1064,132 L1072,165 L1072,165 L1088,166 L1104,168 L1112,149 L1120,168 L1120,170 L1128,151 L1136,170 L1136,173 L1144,146 L1152,173 L1152,177 L1168,180 L1184,184 L1192,156 L1200,184 L1200,187 L1200,260 Z"/>
                        <path class="r3" d="M0,260 L0,223 L6,191 L12,223 L12,224 L24,225 L30,192 L36,225 L36,226 L42,205 L48,226 L48,227 L54,202 L60,227 L60,228 L66,189 L72,228 L72,228 L78,202 L84,228 L84,228 L96,228 L102,207 L108,228 L108,228 L114,207 L120,228 L120,228 L132,227 L138,206 L144,227 L144,226 L150,187 L156,226 L156,225 L162,199 L168,225 L168,223 L174,203 L180,223 L180,222 L186,184 L192,222 L192,221 L198,191 L204,221 L204,219 L216,217 L222,181 L228,217 L228,216 L234,193 L240,216 L240,214 L246,190 L252,214 L252,213 L258,183 L264,213 L264,212 L270,185 L276,212 L276,211 L282,174 L288,211 L288,210 L294,182 L300,210 L300,209 L306,172 L312,209 L312,209 L318,172 L324,209 L324,209 L330,180 L336,209 L336,209 L342,190 L348,209 L348,209 L354,187 L360,209 L360,209 L366,174 L372,209 L372,210 L378,182 L384,210 L384,210 L390,180 L396,210 L396,211 L402,182 L408,211 L408,211 L414,177 L420,211 L420,212 L426,182 L432,212 L432,212 L438,188 L444,212 L444,213 L450,184 L456,213 L456,213 L462,179 L468,213 L468,213 L480,213 L486,182 L492,213 L492,213 L498,184 L504,213 L504,213 L510,185 L516,213 L516,212 L522,184 L528,212 L528,212 L540,211 L546,175 L552,211 L552,210 L564,209 L570,179 L576,209 L576,208 L588,207 L594,186 L600,207 L600,206 L606,179 L612,206 L612,205 L618,182 L624,205 L624,204 L630,172 L636,204 L636,204 L642,167 L648,204 L648,203 L654,170 L660,203 L660,203 L666,182 L672,203 L672,203 L684,203 L696,203 L702,165 L708,203 L708,204 L714,176 L720,204 L720,205 L732,206 L738,184 L744,206 L744,207 L750,178 L756,207 L756,208 L762,186 L768,208 L768,210 L774,177 L780,210 L780,211 L786,182 L792,211 L792,213 L798,195 L804,213 L804,215 L810,183 L816,215 L816,216 L822,197 L828,216 L828,218 L840,219 L846,181 L852,219 L852,220 L858,197 L864,220 L864,222 L870,187 L876,222 L876,222 L882,202 L888,222 L888,223 L894,186 L900,223 L900,224 L906,200 L912,224 L912,224 L918,187 L924,224 L924,224 L930,191 L936,224 L936,224 L942,205 L948,224 L948,224 L954,197 L960,224 L960,223 L966,185 L972,223 L972,223 L978,188 L984,223 L984,222 L990,186 L996,222 L996,221 L1002,185 L1008,221 L1008,221 L1014,195 L1020,221 L1020,220 L1026,182 L1032,220 L1032,219 L1038,198 L1044,219 L1044,219 L1050,196 L1056,219 L1056,218 L1062,197 L1068,218 L1068,218 L1074,196 L1080,218 L1080,218 L1086,193 L1092,218 L1092,218 L1098,194 L1104,218 L1104,218 L1110,196 L1116,218 L1116,218 L1122,200 L1128,218 L1128,219 L1134,200 L1140,219 L1140,219 L1146,190 L1152,219 L1152,220 L1158,192 L1164,220 L1164,221 L1176,222 L1182,186 L1188,222 L1188,222 L1194,194 L1200,222 L1200,223 L1206,197 L1212,223 L1200,260 Z"/></svg>'),
                  div(class = "hero-facts",
                      div(tags$b("4"), "integrated tools"),
                      div(tags$b("19"), "allometric models"),
                      div(tags$b("5"), "machine learning methods"),
                      div(tags$b("17"), "field calculators")
                  )
              ),
              div(class = "modules",
                  div(class = "module",
                      div(class = "module-icon", icon("ruler")),
                      h3("Management & mensuration"),
                      p("Calculators for allowable cut, MAI and CAI, rotation age, Faustmann LEV, basal area and log volume."),
                      div(class = "tags", span("Smalian"), span("Huber"), span("Newton"), span("Quarter girth")),
                      actionButton("go_mgmt", "Open calculators", class = "btn-ghost")
                  ),
                  div(class = "module",
                      div(class = "module-icon", icon("ruler-vertical")),
                      h3("DendroFit ML"),
                      p("Fit height-diameter curves with ten nonlinear models and five ML methods, then validate on held-out trees."),
                      div(class = "tags", span("N\u00e4slund"), span("Curtis"), span("Wykoff"), span("RF, SVM, ANN")),
                      actionButton("go_hd", "Fit H-D models", class = "btn-ghost")
                  ),
                  div(class = "module",
                      div(class = "module-icon", icon("cubes")),
                      h3("DendroVol ML"),
                      p("Predict stem volume from diameter and height. Compare allometric equations with tree ensembles and neural networks."),
                      div(class = "tags", span("Power (H+D)"), span("GBM"), span("Neural net")),
                      actionButton("go_vol", "Predict volume", class = "btn-ghost")
                  ),
                  div(class = "module carbon",
                      div(class = "module-icon", icon("leaf")),
                      h3("TreeCarbonPro"),
                      p("Estimate above- and below-ground biomass and carbon stock for a single tree using standard IPCC methods."),
                      div(class = "tags", span("Chave 2005/2014"), span("IPCC BEF"), span("Root:shoot")),
                      actionButton("go_carbon", "Estimate carbon", class = "btn-ghost")
                  )
              ),
              div(class = "howto",
                  h3("Working with your own data"),
                  div(class = "howto-steps",
                      div(div(class = "n", "1"), h4("Prepare a CSV"),
                          p("One row per tree, with columns for diameter (cm), height (m) and, for volume models, measured volume (m\u00b3).")),
                      div(div(class = "n", "2"), h4("Upload and map columns"),
                          p("Open DendroFit ML or DendroVol ML, upload the file and tell the app which column is which.")),
                      div(div(class = "n", "3"), h4("Run and compare"),
                          p("Pick models, run the analysis and read RMSE, MAE and R\u00b2 on the test split to choose the best fit."))
                  )
              )
      ),
      
      # ============================================================
      # FOREST MANAGEMENT TAB
      # ============================================================
      tabItem(tabName = "management",
              div(class = "page-head",
                  h2("Forest management"),
                  p("Yield, increment and rotation economics. See where your stand sits on each curve."),
                  div(class = "live-pill", "Results update as you type")),
              fluidRow(
                column(width = 6,
                       div(class = "formula-card",
                           h3("Annual Allowable Cut (AAC)"),
                           p(class = "description", "Determines the amount of harvestable volume annually to ensure forest sustainability."),
                           div(class = "math-formula",
                               helpText("$$ AAC = \\frac{GS + MAI \\times \\frac{R}{2}}{R} $$"),
                               p("GS = Growing stock, MAI = Mean Annual Increment, R = Rotation period")
                           ),
                           div(class = "input-section",
                               h4("Calculate AAC"),
                               fluidRow(
                                 column(4, numericInput("aac_gs", "Growing Stock (m³)", 1000, min = 0)),
                                 column(4, numericInput("aac_mai", "MAI (m³/year)", 50, min = 0)),
                                 column(4, numericInput("aac_r", "Rotation (years)", 30, min = 1))
                               ),
                               actionButton("calc_aac", "Calculate AAC", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("aac_result"),
                               plotlyOutput("aac_plot", height = "220px")
                           )
                       )
                ),
                column(width = 6,
                       div(class = "formula-card",
                           h3("Mean Annual Increment (MAI)"),
                           p(class = "description", "Average annual growth over the lifetime of a stand."),
                           div(class = "math-formula",
                               helpText("$$ MAI = \\frac{\\text{Total Volume}}{\\text{Age}} $$")
                           ),
                           div(class = "input-section",
                               h4("Calculate MAI"),
                               fluidRow(
                                 column(6, numericInput("mai_vol", "Total Volume (m³)", 600, min = 0)),
                                 column(6, numericInput("mai_age", "Stand Age (years)", 20, min = 1))
                               ),
                               actionButton("calc_mai", "Calculate MAI", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("mai_result"),
                               plotlyOutput("mai_plot", height = "220px")
                           )
                       )
                )
              ),
              fluidRow(
                column(width = 6,
                       div(class = "formula-card",
                           h3("Current Annual Increment (CAI)"),
                           p(class = "description", "Growth in the most recent year."),
                           div(class = "math-formula",
                               helpText("$$ CAI = V_t - V_{t-1} $$")
                           ),
                           div(class = "input-section",
                               h4("Calculate CAI"),
                               fluidRow(
                                 column(6, numericInput("cai_v_current", "Current Volume (m³)", 650, min = 0)),
                                 column(6, numericInput("cai_v_prev", "Previous Volume (m³)", 600, min = 0))
                               ),
                               actionButton("calc_cai", "Calculate CAI", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("cai_result"),
                               plotlyOutput("cai_vs_mai_plot", height = "220px")
                           )
                       )
                ),
                column(width = 6,
                       div(class = "formula-card",
                           h3("Land Expectation Value (Faustmann)"),
                           p(class = "description", "Present value of infinite series of forest rotations."),
                           div(class = "math-formula",
                               helpText("$$ LEV = \\frac{R_t - C_0(1+i)^t}{(1+i)^t - 1} $$"),
                               p("R_t = Revenue, C_0 = Establishment costs, i = Interest rate, t = Rotation")
                           ),
                           div(class = "input-section",
                               h4("Calculate LEV"),
                               fluidRow(
                                 column(4, numericInput("lev_rev", "Revenue (Rs)", 500000, min = 0)),
                                 column(4, numericInput("lev_cost", "Est. Costs (Rs)", 100000, min = 0)),
                                 column(4, numericInput("lev_interest", "Interest Rate (%)", 5, min = 0.1, max = 25, step = 0.1))
                               ),
                               numericInput("lev_rotation", "Rotation Age (years)", 30, min = 1),
                               actionButton("calc_lev", "Calculate LEV", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("lev_result"),
                               plotlyOutput("lev_plot", height = "220px")
                           )
                       )
                )
              ),
              fluidRow(
                column(width = 6,
                       div(class = "formula-card",
                           h3("Normal Growing Stock (NGS)"),
                           p(class = "description", "Ideal volume in a fully regulated forest."),
                           div(class = "math-formula",
                               helpText("$$ NGS = \\frac{I \\times R}{2} $$"),
                               p("I = Annual increment, R = Rotation age")
                           ),
                           div(class = "input-section",
                               h4("Calculate NGS"),
                               fluidRow(
                                 column(6, numericInput("ngs_increment", "Annual Increment (m³/year)", 50, min = 0)),
                                 column(6, numericInput("ngs_rotation", "Rotation Age (years)", 30, min = 1))
                               ),
                               actionButton("calc_ngs", "Calculate NGS", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("ngs_result"),
                               plotlyOutput("ngs_plot", height = "220px")
                           )
                       )
                ),
                column(width = 6,
                       div(class = "formula-card",
                           h3("Rotation Age Simulator"),
                           p(class = "description", "Visualize where MAI = CAI (optimal rotation age)."),
                           div(class = "math-formula",
                               helpText("Optimal rotation when: $$ MAI = CAI $$")
                           ),
                           div(class = "input-section",
                               h4("Simulate Growth Curves"),
                               fluidRow(
                                 column(6, numericInput("growth_rate", "Growth Rate Param", 0.15, min = 0.01, max = 0.5, step = 0.01)),
                                 column(6, numericInput("max_age", "Max Age to Plot", 100, min = 10, max = 200))
                               ),
                               actionButton("calc_rotation", "Simulate", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("rotation_result"),
                               plotlyOutput("rotation_plot", height = "220px")
                           )
                       )
                )
              )
      ),
      
      # ============================================================
      # FOREST MENSURATION TAB
      # ============================================================
      tabItem(tabName = "mensuration",
              div(class = "page-head",
                  h2("Forest mensuration"),
                  p("Tree and log measurement formulas. Each volume method feeds the comparison at the bottom of the page."),
                  div(class = "live-pill", "Results update as you type")),
              fluidRow(
                column(width = 6,
                       div(class = "formula-card",
                           h3("1. DBH from Girth"),
                           p(class = "description", "Diameter at breast height from girth measurement."),
                           div(class = "math-formula", helpText("$$ DBH = \\frac{\\text{Girth}}{\\pi} $$")),
                           div(class = "input-section",
                               numericInput("dbh_girth", "Girth at BH (cm)", 100, min = 0),
                               actionButton("calc_dbh", "Calculate DBH", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("dbh_result"),
                               plotlyOutput("dbh_viz", height = "200px")
                           )
                       )
                ),
                column(width = 6,
                       div(class = "formula-card",
                           h3("2. Basal Area (BA)"),
                           p(class = "description", "Cross-sectional area of stem at breast height."),
                           div(class = "math-formula", helpText("$$ BA = 0.7854 \\times D^2 $$")),
                           div(class = "input-section",
                               numericInput("ba_dbh", "DBH (cm)", 30, min = 0),
                               actionButton("calc_ba", "Calculate BA", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("ba_result"),
                               plotlyOutput("ba_viz", height = "200px")
                           )
                       )
                )
              ),
              fluidRow(
                column(width = 6,
                       div(class = "formula-card",
                           h3("3. Basal Area per Hectare"),
                           div(class = "math-formula", helpText("$$ BA_{ha} = \\frac{\\sum BA_{tree}}{\\text{Plot Area}} \\times 10000 $$")),
                           div(class = "input-section",
                               fluidRow(
                                 column(6, numericInput("ba_ha_sum", "Sum of BAs in Plot (m²)", 1.5, min = 0, step = 0.1)),
                                 column(6, numericInput("ba_ha_plot", "Plot Area (m²)", 500, min = 1))
                               ),
                               actionButton("calc_ba_ha", "Calculate BA/ha", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("ba_ha_result"),
                               plotlyOutput("ba_ha_viz", height = "200px")
                           )
                       )
                ),
                column(width = 6,
                       div(class = "formula-card",
                           h3("4. Tree Volume (General)"),
                           div(class = "math-formula",
                               helpText("$$ V = BA \\times H \\times F $$"),
                               p("F = Form Factor")
                           ),
                           div(class = "input-section",
                               fluidRow(
                                 column(4, numericInput("vol_ba", "Basal Area (m²)", 0.07, min = 0, step = 0.01)),
                                 column(4, numericInput("vol_height", "Height (m)", 20, min = 0)),
                                 column(4, numericInput("vol_form", "Form Factor", 0.5, min = 0, max = 1, step = 0.05))
                               ),
                               actionButton("calc_vol", "Calculate Volume", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("vol_result"),
                               plotlyOutput("vol_viz", height = "200px")
                           )
                       )
                )
              ),
              fluidRow(
                column(width = 6,
                       div(class = "formula-card",
                           h3("5. Smalian's Formula"),
                           div(class = "math-formula", helpText("$$ V = \\frac{(A_1 + A_2)}{2} \\times L $$")),
                           div(class = "input-section",
                               fluidRow(
                                 column(4, numericInput("smalian_a1", "Large End Area (m²)", 0.08, min = 0, step = 0.01)),
                                 column(4, numericInput("smalian_a2", "Small End Area (m²)", 0.05, min = 0, step = 0.01)),
                                 column(4, numericInput("smalian_l", "Log Length (m)", 3, min = 0))
                               ),
                               actionButton("calc_smalian", "Calculate", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("smalian_result"),
                               plotlyOutput("smalian_viz", height = "200px")
                           )
                       )
                ),
                column(width = 6,
                       div(class = "formula-card",
                           h3("6. Huber's Formula"),
                           div(class = "math-formula", helpText("$$ V = A_m \\times L $$")),
                           div(class = "input-section",
                               fluidRow(
                                 column(6, numericInput("huber_am", "Mid-length Area (m²)", 0.065, min = 0, step = 0.01)),
                                 column(6, numericInput("huber_l", "Log Length (m)", 3, min = 0))
                               ),
                               actionButton("calc_huber", "Calculate", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("huber_result"),
                               plotlyOutput("huber_viz", height = "200px")
                           )
                       )
                )
              ),
              fluidRow(
                column(width = 6,
                       div(class = "formula-card",
                           h3("7. Newton's Formula"),
                           div(class = "math-formula", helpText("$$ V = \\frac{(A_1 + 4A_m + A_2)}{6} \\times L $$")),
                           div(class = "input-section",
                               fluidRow(
                                 column(4, numericInput("newton_a1", "Large End (m²)", 0.08, min = 0, step = 0.01)),
                                 column(4, numericInput("newton_am", "Mid-length (m²)", 0.065, min = 0, step = 0.01)),
                                 column(4, numericInput("newton_a2", "Small End (m²)", 0.05, min = 0, step = 0.01))
                               ),
                               numericInput("newton_l", "Log Length (m)", 3, min = 0),
                               actionButton("calc_newton", "Calculate", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("newton_result"),
                               plotlyOutput("newton_viz", height = "200px")
                           )
                       )
                ),
                column(width = 6,
                       div(class = "formula-card",
                           h3("8. Form Quotient"),
                           div(class = "math-formula", helpText("$$ FQ = \\frac{\\text{Diameter at Half Height}}{DBH} $$")),
                           div(class = "input-section",
                               fluidRow(
                                 column(6, numericInput("fq_dbh", "DBH (cm)", 30, min = 0)),
                                 column(6, numericInput("fq_dh", "Diameter @ Half Height (cm)", 20, min = 0))
                               ),
                               actionButton("calc_fq", "Calculate FQ", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("fq_result"),
                               plotlyOutput("fq_viz", height = "200px")
                           )
                       )
                )
              ),
              fluidRow(
                column(width = 6,
                       div(class = "formula-card",
                           h3("9. Quarter Girth Formula"),
                           div(class = "math-formula", helpText("$$ Volume = \\frac{G^2 \\times L}{40000} $$")),
                           div(class = "input-section",
                               fluidRow(
                                 column(6, numericInput("qg_girth", "Girth (cm)", 100, min = 0)),
                                 column(6, numericInput("qg_length", "Log Length (m)", 6, min = 0))
                               ),
                               actionButton("calc_qg", "Calculate", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("qg_result"),
                               plotlyOutput("qg_viz", height = "200px")
                           )
                       )
                ),
                column(width = 6,
                       div(class = "formula-card",
                           h3("10. Approx. Volume (D & H)"),
                           div(class = "math-formula", helpText("$$ V = \\frac{D^2 \\times H}{1000} $$")),
                           div(class = "input-section",
                               fluidRow(
                                 column(6, numericInput("approx_dbh", "DBH (cm)", 30, min = 0)),
                                 column(6, numericInput("approx_height", "Height (m)", 20, min = 0))
                               ),
                               actionButton("calc_approx", "Calculate", class = "calculate-btn")
                           ),
                           div(class = "result-display",
                               verbatimTextOutput("approx_result"),
                               plotlyOutput("approx_viz", height = "200px")
                           )
                       )
                )
              ),
              fluidRow(
                box(width = 12, title = "How the volume methods compare", status = "primary", solidHeader = TRUE,
                    fluidRow(
                      column(5, DTOutput("volume_comparison_table")),
                      column(7, plotlyOutput("volume_comparison_plot", height = "300px"))
                    )
                )
              ),
              fluidRow(
                box(width = 12, title = "Multi-Tree Calculator", status = "info", solidHeader = TRUE,
                    fluidRow(
                      column(4, numericInput("multi_num_trees", "Number of Trees", 5, min = 1, max = 20)),
                      column(4, selectInput("multi_formula", "Formula", choices = c("General Formula" = "general", "Quick Approximation" = "approx"))),
                      column(4, br(), actionButton("generate_tree_inputs", "Generate Inputs", class = "calculate-btn"))
                    ),
                    br(),
                    uiOutput("multi_tree_inputs"),
                    br(),
                    actionButton("calc_multi_trees", "Calculate All Trees", class = "calculate-btn"),
                    br(), br(),
                    DTOutput("multi_tree_results"),
                    plotlyOutput("multi_tree_viz", height = "300px")
                )
              )
      ),
      
      # ---- References Tab ----
      tabItem(tabName = "references",
              fluidRow(
                box(width = 12, title = "Academic References", status = "primary", solidHeader = TRUE,
                    tags$ul(
                      tags$li(strong("Avery, T. E., & Burkhart, H. E. (2015)."), " Forest Measurements (5th ed.). Waveland Press."),
                      tags$li(strong("Philip, M. S. (1994)."), " Measuring Trees and Forests. CAB International."),
                      tags$li(strong("Chaturvedi, A. N., & Khanna, L. S. (1982)."), " Forest Mensuration. International Book Distributors."),
                      tags$li(strong("Smith, D. M. (1986)."), " The Practice of Silviculture. Wiley."),
                      tags$li(strong("FAO. (1998)."), " Forest Resources Assessment 1990. FAO Forestry Paper 140."),
                      tags$li(strong("Van Laar, A., & Akça, A. (2007)."), " Forest Mensuration. Springer."),
                      tags$li(strong("Burkhart, H. E., & Tomé, M. (2012)."), " Modeling Forest Trees and Stands. Springer."),
                      tags$li(strong("Chave et al. (2005)."), " Tree allometry and improved estimation of carbon stocks. Oecologia, 145(1), 87–99."),
                      tags$li(strong("Chave et al. (2014)."), " Improved allometric models to estimate aboveground biomass. Global Change Biology, 20(10), 3177–3190."),
                      tags$li(strong("IPCC (2006)."), " Guidelines for National Greenhouse Gas Inventories, Vol. 4."),
                      tags$li(strong("Mokany et al. (2006)."), " Critical analysis of root:shoot ratios. Global Change Biology, 12(1), 84–96.")
                    )
                )
              ),
              fluidRow(
                box(width = 6, title = "Web Resources", status = "info", solidHeader = TRUE,
                    tags$ul(
                      tags$li(a("FAO Forestry", href = "https://www.fao.org/forestry", target = "_blank")),
                      tags$li(a("USDA Forest Service", href = "https://www.fs.usda.gov/", target = "_blank")),
                      tags$li(a("SKUAST-Kashmir", href = "https://www.skuastkashmir.ac.in/", target = "_blank")),
                      tags$li(a("Forest Research Institute", href = "https://fri.icfre.gov.in/", target = "_blank"))
                    )
                ),
                box(width = 6, title = "Recommended Journals", status = "success", solidHeader = TRUE,
                    tags$ul(
                      tags$li("Forest Ecology and Management"),
                      tags$li("Journal of Forestry"),
                      tags$li("Forest Science"),
                      tags$li("Canadian Journal of Forest Research"),
                      tags$li("Annals of Forest Science"),
                      tags$li("European Journal of Forest Research")
                    )
                )
              )
      ),
      
      # ============================================================
      # DENDROFIT ML TABS
      # ============================================================
      tabItem(tabName = "hd_dashboard",
              fluidRow(
                valueBoxOutput("hd_totalObsBox", width = 3),
                valueBoxOutput("hd_bestModelBox", width = 3),
                valueBoxOutput("hd_bestMLModelBox", width = 3),
                valueBoxOutput("hd_accuracyBox", width = 3)
              ),
              fluidRow(
                box(title = "Height vs. Diameter", status = "success", solidHeader = TRUE, width = 12,
                    plotlyOutput("hd_dashboardPlot", height = "400px")
                )
              ),
              fluidRow(
                box(title = "Quick Stats", status = "success", solidHeader = TRUE, width = 4,
                    verbatimTextOutput("hd_quickStats")),
                box(title = "Best Traditional Model", status = "success", solidHeader = TRUE, width = 4,
                    DTOutput("hd_quickModelStats")),
                box(title = "Best ML Model", status = "success", solidHeader = TRUE, width = 4,
                    DTOutput("hd_quickMLModelStats"))
              )
      ),
      tabItem(tabName = "hd_data",
              fluidRow(
                box(title = "Data Summary", status = "success", solidHeader = TRUE, width = 12,
                    verbatimTextOutput("hd_dataSummary")),
                box(title = "Data Preview", status = "success", solidHeader = TRUE, width = 12,
                    DTOutput("hd_dataPreview")),
                box(title = "Scatter Plot", status = "success", solidHeader = TRUE, width = 12,
                    plotlyOutput("hd_scatterPlot", height = "400px"))
              )
      ),
      tabItem(tabName = "hd_models",
              fluidRow(
                box(title = "Traditional Model Comparison", status = "success", solidHeader = TRUE, width = 12,
                    DTOutput("hd_modelComparison")),
                box(title = "Model Parameters", status = "success", solidHeader = TRUE, width = 12,
                    verbatimTextOutput("hd_modelParams")),
                box(title = "Models Visualization", status = "success", solidHeader = TRUE, width = 12,
                    plotlyOutput("hd_modelPlot", height = "500px"))
              )
      ),
      tabItem(tabName = "hd_ml_models",
              fluidRow(
                box(title = "ML Model Comparison", status = "success", solidHeader = TRUE, width = 12,
                    DTOutput("hd_mlModelComparison")),
                tabBox(title = "ML Model Details", width = 12, id = "hd_ml_tabs",
                       tabPanel("Decision Tree", plotOutput("hd_dtPlot", height = "400px"), verbatimTextOutput("hd_dtSummary")),
                       tabPanel("Random Forest", plotOutput("hd_rfPlot", height = "400px"), verbatimTextOutput("hd_rfSummary")),
                       tabPanel("SVM", verbatimTextOutput("hd_svmSummary")),
                       tabPanel("Gradient Boosting", plotOutput("hd_gbmPlot", height = "400px"), verbatimTextOutput("hd_gbmSummary")),
                       tabPanel("Neural Network (ANN)",
                                plotOutput("hd_annPlot", height = "400px"),
                                verbatimTextOutput("hd_annSummary"))
                ),
                box(title = "ML Models Predictions", status = "success", solidHeader = TRUE, width = 12,
                    plotlyOutput("hd_mlModelPlot", height = "500px"))
              )
      ),
      tabItem(tabName = "hd_validation",
              fluidRow(
                tabBox(title = "Cross-Validation Results", width = 12, id = "hd_cv_tabs",
                       tabPanel("Training Data",
                                fluidRow(
                                  column(6, h4("Traditional Models - Training"), DTOutput("hd_trainResults")),
                                  column(6, h4("ML Models - Training"), DTOutput("hd_mlTrainResults"))
                                )
                       ),
                       tabPanel("Testing Data",
                                fluidRow(
                                  column(6, h4("Traditional Models - Testing"), DTOutput("hd_testResults")),
                                  column(6, h4("ML Models - Testing"), DTOutput("hd_mlTestResults"))
                                )
                       ),
                       tabPanel("Combined Results",
                                h4("All Models - Testing Data"),
                                DTOutput("hd_combinedResults")
                       )
                ),
                box(title = "Validation Plot", status = "success", solidHeader = TRUE, width = 12,
                    plotlyOutput("hd_validationPlot", height = "500px"))
              )
      ),
      
      # ============================================================
      # DENDROVOL ML TAB
      # ============================================================
      tabItem(tabName = "DendroVolML",
              div(class = "page-head",
                  h2("DendroVol ML"),
                  p("Predict stem volume from diameter and height with allometric equations and machine learning. A 10-tree demo dataset is loaded until you upload your own CSV.")
              ),
              fluidRow(
                column(3,
                       div(class = "well",
                           fileInput("dv_datafile", "Upload CSV File", accept = ".csv"),
                           hr(),
                           selectInput("dv_diameter_var", "Diameter Variable", choices = NULL),
                           selectInput("dv_height_var", "Height Variable", choices = NULL),
                           selectInput("dv_volume_var", "Volume Variable", choices = NULL),
                           selectInput("dv_split", "Data Split",
                                       choices = c("50:50", "70:30", "80:20"), selected = "70:30"),
                           hr(),
                           h4("Traditional Models"),
                           div(style = "max-height: 180px; overflow-y: auto;",
                               checkboxGroupInput("dv_allometric_models", NULL,
                                                  choices = c("Linear (D)" = "Linear.D1", "Logarithmic (D)" = "Logarithmic.D2",
                                                              "Inverse (D)" = "Inverse.D3", "Quadratic (D)" = "Quadratic.D4",
                                                              "Cubic (D)" = "Cubic.D5", "Power (D)" = "Power.D7",
                                                              "Exponential (D)" = "Exponential.D8",
                                                              "Linear (H+D)" = "Linear.HD1", "Power (H+D)" = "Power.HD7"),
                                                  selected = c("Linear.D1", "Power.D7", "Linear.HD1"))
                           ),
                           hr(),
                           h4("ML Models"),
                           checkboxGroupInput("dv_ml_models", NULL,
                                              choices = c("Decision Tree" = "DT", "Random Forest" = "RF",
                                                          "SVM" = "SVM", "Gradient Boosting" = "GB",
                                                          "Neural Network (ANN)" = "ANN"),
                                              selected = c("DT", "RF")),
                           hr(),
                           h4("ANN Settings"),
                           numericInput("dv_ann_hidden1", "Hidden Neurons (Layer 1):", 8, min = 2, max = 30),
                           numericInput("dv_ann_hidden2", "Hidden Neurons (Layer 2):", 4, min = 0, max = 20),
                           numericInput("dv_ann_thresh",  "ANN Threshold:", 0.05, min = 0.01, max = 0.5, step = 0.01),
                           hr(),
                           actionButton("dv_run", "Run volume analysis", icon = icon("play"), class = "btn-forest btn-block"),
                           hr(),
                           h4("Make Predictions"),
                           numericInput("dv_pred_diameter", "Diameter:", value = 20),
                           numericInput("dv_pred_height", "Height:", value = 15),
                           actionButton("dv_predict", "Predict volume", icon = icon("ruler-combined"), class = "btn-larch btn-block"),
                           hr(),
                           downloadButton("dv_download_results", "Download results (CSV)", class = "btn-ghost btn-block")
                       )
                ),
                column(9,
                       tabsetPanel(
                         tabPanel("Dashboard",
                                  fluidRow(
                                    column(12,
                                           h3("Model Performance Summary"),
                                           verbatimTextOutput("dv_best_models"),
                                           plotOutput("dv_results_plot", height = "400px")
                                    )
                                  ),
                                  fluidRow(
                                    column(12,
                                           h3("Predicted vs Observed Volume"),
                                           plotOutput("dv_scatter_plot", height = "400px")
                                    )
                                  ),
                                  fluidRow(
                                    column(12,
                                           h3("Prediction Results"),
                                           div(id = "dv_prediction_results", verbatimTextOutput("dv_prediction_results"))
                                    )
                                  )
                         ),
                         tabPanel("Model Details",
                                  tabsetPanel(
                                    tabPanel("Allometric Models", verbatimTextOutput("dv_allometric_model_summaries")),
                                    tabPanel("ML Models", verbatimTextOutput("dv_ml_model_info"), plotOutput("dv_ml_importance_plot", height = "400px"))
                                  )
                         ),
                         tabPanel("Data",
                                  h3("Dataset Preview"),
                                  DTOutput("dv_data_preview")
                         )
                       )
                )
              )
      ),
      
      # ============================================================
      # TREECARBONPRO TABS
      # ============================================================
      tabItem(tabName = "carbon_input",
              fluidRow(
                box(width = 12, solidHeader = TRUE, status = "primary", title = "TreeCarbonPro — Key Equations",
                    div(
                      h4("Allometric Equations & Models Used:"),
                      div(class = "equation-box",
                          p(class = "ref-title", "Chave et al. (2005)"),
                          p(class = "math", "AGB = 0.0509 × ρ × DBH² × H")
                      ),
                      div(class = "equation-box",
                          p(class = "ref-title", "Chave et al. (2014)"),
                          p(class = "math", "AGB = 0.0673 × (ρ × DBH² × H)^0.976")
                      ),
                      div(class = "equation-box",
                          p(class = "ref-title", "IPCC (2006)"),
                          p(class = "math", "AGB = Volume × Wood Density × BEF"),
                          p(class = "math", "Carbon Stock = Total Biomass × Carbon Fraction")
                      ),
                      div(class = "equation-box",
                          p(class = "ref-title", "Mokany et al. (2006)"),
                          p(class = "math", "BGB = AGB × R:S"),
                          p(class = "math", "Total Biomass = AGB + BGB")
                      ),
                      p(class = "hint", icon("arrow-left"), " Enter tree parameters in the sidebar, then press Estimate carbon stock to open the results.")
                    )
                )
              )
      ),
      tabItem(tabName = "carbon_results",
              div(class = "page-head",
                  h2("Carbon stock of a single tree"),
                  p("Change any value in the sidebar and the results recalculate."),
                  div(class = "live-pill", "Live")),
              fluidRow(
                valueBoxOutput("tc_vb_agb", width = 3),
                valueBoxOutput("tc_vb_bgb", width = 3),
                valueBoxOutput("tc_vb_total", width = 3),
                valueBoxOutput("tc_vb_carbon", width = 3)
              ),
              fluidRow(
                box(width = 7, solidHeader = TRUE, status = "success", title = "Where the biomass comes from",
                    plotlyOutput("tc_bar", height = "330px")),
                box(width = 5, solidHeader = TRUE, status = "success", title = "Above vs below ground",
                    plotlyOutput("tc_donut", height = "330px"))
              ),
              fluidRow(
                box(width = 12, solidHeader = TRUE, status = "primary", title = "Detailed results",
                    verbatimTextOutput("tc_results"),
                    downloadButton("tc_downloadResults", "Download results (CSV)", class = "btn-forest"),
                    br(), br(),
                    h4("References"),
                    tags$ul(class = "ref-list",
                            tags$li("Chave et al. (2005). Tree allometry. Oecologia, 145(1), 87–99."),
                            tags$li("Chave et al. (2014). Improved allometric models. Global Change Biology, 20(10), 3177–3190."),
                            tags$li("IPCC (2006). Guidelines for National GHG Inventories, Vol. 4."),
                            tags$li("Mokany et al. (2006). Root:shoot ratios. Global Change Biology, 12(1), 84–96.")
                    )
                )
              )
      ),
      
      # ============================================================
      # ABOUT TAB
      # ============================================================
      tabItem(tabName = "about",
              fluidRow(
                box(width = 12, solidHeader = TRUE, status = "success",
                    title = "About ImForestSuite",
                    div(class = "about-section",
                        div(class = "about-lead",
                            h3("A unified forestry analytics platform"),
                            p("Mensuration, growth modelling, volume prediction and carbon accounting in one place, built for foresters, researchers and students.")
                        ),
                        hr(),
                        h4("Developed by:"),
                        h4(strong("M. Iqbal Jeelani "), style = "font-weight:bold;"),
                        p("Sher-e-Kashmir University of Agricultural Sciences and Technology of Kashmir (SKUAST-K)"),
                        p(icon("envelope"), " Email: ", a("jeelani.miqbal@gmail.com", href = "mailto:jeelani.miqbal@gmail.com")),
                        p(icon("youtube"), " YouTube: ", a("@Iqbalstat", href = "https://www.youtube.com/@Iqbalstat", target = "_blank")),
                        hr(),
                        h4("Modules Included:"),
                        tags$ul(
                          tags$li(strong("Forest Management & Mensuration (ForAMat):"), " AAC, MAI, CAI, LEV (Faustmann), NGS, DBH, BA, Smalian, Huber, Newton, Form Quotient, Quarter Girth, Multi-tree calculator."),
                          tags$li(strong("DendroFit ML:"), " Height-diameter modeling with 10 traditional nonlinear models (Naslund, Curtis, Michailoff, Power, Wykoff, Michaelis-Menten) and 5 ML algorithms (RF, SVM, GBM, DT, ANN) with cross-validation."),
                          tags$li(strong("DendroVol ML:"), " Tree volume prediction using 9 allometric models (diameter-based and H+D-based) plus 5 ML models (DT, RF, SVM, GB, ANN)."),
                          tags$li(strong("TreeCarbonPro:"), " Biomass and carbon stock estimation using Chave et al. (2005, 2014), IPCC BEF method, and Mokany root-to-shoot ratios.")
                        ),
                        hr(),
                        p(style = "text-align:center; font-style:italic;", "Built with R Shiny, shinydashboard, ggplot2, plotly, randomForest, e1071, gbm, rpart, mgcv, nls2, and neuralnet.")
                    )
                )
              )
      )
    ) # end tabItems
  ) # end dashboardBody
) # end dashboardPage


# ============================================================
# SERVER
# ============================================================
server <- function(input, output, session) {
  
  # ---- Home navigation ----
  observeEvent(input$go_mgmt,   updateTabItems(session, "main_sidebar", "management"))
  observeEvent(input$go_hd,     updateTabItems(session, "main_sidebar", "hd_data"))
  observeEvent(input$go_vol,    updateTabItems(session, "main_sidebar", "DendroVolML"))
  observeEvent(input$go_carbon, updateTabItems(session, "main_sidebar", "carbon_input"))
  
  # ============================================================
  # FOREST MANAGEMENT SERVER
  # ============================================================
  fm_results <- reactiveValues(
    mgmt = data.frame(Parameter = character(), Value = numeric(), stringsAsFactors = FALSE)
  )
  
  observeEvent(input$calc_aac, {
    gs <- input$aac_gs; mai <- input$aac_mai; r <- input$aac_r
    aac <- (gs + mai * (r / 2)) / r
    output$aac_result <- renderText(paste("Annual Allowable Cut (AAC) =", round(aac, 2), "m³/year"))
    r_vals <- seq(5, 2 * r, by = 5)
    aac_vals <- (gs + mai * (r_vals / 2)) / r_vals
    output$aac_plot <- renderPlotly({
      ggplotly(ggplot(data.frame(R = r_vals, AAC = aac_vals), aes(x = R, y = AAC)) +
                 geom_line(color = "#1d4a3a", size = 1.2) +
                 geom_vline(xintercept = r, linetype = "dashed", color = "#d9a21b") +
                 labs(x = "Rotation (years)", y = "AAC (m³/year)", title = "AAC vs. Rotation") +
                 theme_forest())
    })
  })
  
  observeEvent(input$calc_mai, {
    mai <- input$mai_vol / input$mai_age
    output$mai_result <- renderText(paste("MAI =", round(mai, 2), "m³/year"))
    ages <- seq(1, 2 * input$mai_age)
    vols <- 100 * (1 - exp(-0.05 * ages))^1.5
    mai_seq <- vols / ages
    output$mai_plot <- renderPlotly({
      ggplotly(ggplot(data.frame(Age = ages, MAI = mai_seq), aes(x = Age, y = MAI)) +
                 geom_line(color = "#2f6b52", size = 1.2) +
                 geom_vline(xintercept = input$mai_age, linetype = "dashed", color = "#d9a21b") +
                 labs(x = "Age (years)", y = "MAI (m³/year)", title = "MAI vs. Age") + theme_forest())
    })
  })
  
  observeEvent(input$calc_cai, {
    cai <- input$cai_v_current - input$cai_v_prev
    output$cai_result <- renderText(paste("CAI =", round(cai, 2), "m³/year"))
    output$cai_vs_mai_plot <- renderPlotly({
      ages <- 1:60; vols <- 100 * (1 - exp(-0.05 * ages))^1.5
      mai_seq <- vols / ages
      cai_seq <- c(NA, diff(vols))
      df <- data.frame(Age = ages, MAI = mai_seq, CAI = cai_seq) %>%
        pivot_longer(cols = c(MAI, CAI), names_to = "Type", values_to = "Value")
      ggplotly(ggplot(df, aes(x = Age, y = Value, color = Type)) +
                 geom_line(size = 1.1) +
                 labs(x = "Age", y = "Increment (m³/year)", title = "MAI vs. CAI") + theme_forest())
    })
  })
  
  observeEvent(input$calc_lev, {
    rev <- input$lev_rev; cost <- input$lev_cost
    i <- input$lev_interest / 100; t <- input$lev_rotation
    lev <- (rev - cost * (1 + i)^t) / ((1 + i)^t - 1)
    output$lev_result <- renderText(paste("Land Expectation Value (LEV) = Rs", round(lev, 2)))
    t_vals <- seq(10, 80, by = 5)
    lev_vals <- (rev - cost * (1 + i)^t_vals) / ((1 + i)^t_vals - 1)
    output$lev_plot <- renderPlotly({
      ggplotly(ggplot(data.frame(Rotation = t_vals, LEV = lev_vals), aes(x = Rotation, y = LEV)) +
                 geom_line(color = "#3d5f7a", size = 1.2) +
                 geom_vline(xintercept = t, linetype = "dashed", color = "#d9a21b") +
                 labs(x = "Rotation (years)", y = "LEV (Rs)", title = "LEV vs. Rotation") + theme_forest())
    })
  })
  
  observeEvent(input$calc_ngs, {
    ngs <- (input$ngs_increment * input$ngs_rotation) / 2
    output$ngs_result <- renderText(paste("Normal Growing Stock (NGS) =", round(ngs, 2), "m³"))
    r_seq <- seq(10, 100, by = 5)
    output$ngs_plot <- renderPlotly({
      ggplotly(ggplot(data.frame(R = r_seq, NGS = (input$ngs_increment * r_seq) / 2), aes(x = R, y = NGS)) +
                 geom_line(color = "#a3612a", size = 1.2) +
                 geom_vline(xintercept = input$ngs_rotation, linetype = "dashed", color = "#d9a21b") +
                 labs(x = "Rotation (years)", y = "NGS (m³)", title = "NGS vs. Rotation") + theme_forest())
    })
  })
  
  observeEvent(input$calc_rotation, {
    gr <- input$growth_rate; max_a <- input$max_age
    ages <- 1:max_a
    vols <- 100 * (1 - exp(-gr * ages))^1.5
    mai_v <- vols / ages
    cai_v <- c(NA, diff(vols))
    idx <- which.min(abs(mai_v - cai_v))
    output$rotation_result <- renderText(paste("Approximate optimal rotation age:", idx, "years (where MAI ≈ CAI)"))
    df <- data.frame(Age = ages, MAI = mai_v, CAI = cai_v) %>%
      pivot_longer(cols = c(MAI, CAI), names_to = "Metric", values_to = "Value")
    output$rotation_plot <- renderPlotly({
      ggplotly(ggplot(df, aes(x = Age, y = Value, color = Metric)) +
                 geom_line(size = 1.1, na.rm = TRUE) +
                 geom_vline(xintercept = idx, linetype = "dashed", color = "#2c3e50") +
                 labs(x = "Age (years)", y = "Increment (m³/year)", title = "MAI and CAI Curves") + theme_forest())
    })
  })
  
  # ============================================================
  # FOREST MENSURATION SERVER
  # ============================================================
  mens_results <- reactiveValues(
    volumes = data.frame(Method = character(), Volume = numeric(), stringsAsFactors = FALSE)
  )
  # keep one row per method so live recalculation updates rather than piles up
  mens_upsert <- function(method, vol) {
    df <- isolate(mens_results$volumes)
    df <- df[df$Method != method, , drop = FALSE]
    mens_results$volumes <- rbind(df, data.frame(Method = method, Volume = vol, stringsAsFactors = FALSE))
  }
  
  observeEvent(input$calc_dbh, {
    dbh <- input$dbh_girth / pi
    output$dbh_result <- renderText(paste("DBH =", round(dbh, 2), "cm"))
    girths <- seq(50, 200, by = 10)
    output$dbh_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(Girth = girths, DBH = girths / pi), aes(x = Girth, y = DBH)) +
                 geom_line(color = "#2f6b52", size = 1.1) +
                 geom_point(data = data.frame(Girth = input$dbh_girth, DBH = dbh), color = "#d9a21b", size = 4) +
                 labs(x = "Girth (cm)", y = "DBH (cm)", title = "DBH vs. Girth") + theme_forest())
    })
  })
  
  observeEvent(input$calc_ba, {
    ba <- 0.7854 * (input$ba_dbh / 100)^2
    output$ba_result <- renderText(paste("Basal Area =", round(ba, 5), "m²"))
    d_seq <- seq(5, 80, by = 5)
    output$ba_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(DBH = d_seq, BA = 0.7854 * (d_seq / 100)^2), aes(x = DBH, y = BA)) +
                 geom_line(color = "#1d4a3a", size = 1.1) +
                 geom_point(data = data.frame(DBH = input$ba_dbh, BA = ba), color = "#d9a21b", size = 4) +
                 labs(x = "DBH (cm)", y = "Basal Area (m²)", title = "BA vs. DBH") + theme_forest())
    })
  })
  
  observeEvent(input$calc_ba_ha, {
    ba_ha <- (input$ba_ha_sum / input$ba_ha_plot) * 10000
    output$ba_ha_result <- renderText(paste("Basal Area per Hectare =", round(ba_ha, 2), "m²/ha"))
    plots <- seq(100, 1000, by = 100)
    output$ba_ha_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(Plot = plots, BA = (input$ba_ha_sum / plots) * 10000), aes(x = Plot, y = BA)) +
                 geom_line(color = "#4f8f74", size = 1.1) +
                 geom_point(data = data.frame(Plot = input$ba_ha_plot, BA = ba_ha), color = "#d9a21b", size = 4) +
                 labs(x = "Plot Area (m²)", y = "BA/ha (m²/ha)", title = "BA/ha vs. Plot Size") + theme_forest())
    })
  })
  
  observeEvent(input$calc_vol, {
    vol <- input$vol_ba * input$vol_height * input$vol_form
    output$vol_result <- renderText(paste("Tree Volume =", round(vol, 4), "m³"))
    h_seq <- seq(5, 40, by = 5)
    output$vol_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(H = h_seq, V = input$vol_ba * h_seq * input$vol_form), aes(x = H, y = V)) +
                 geom_line(color = "#3d5f7a", size = 1.1) +
                 geom_point(data = data.frame(H = input$vol_height, V = vol), color = "#d9a21b", size = 4) +
                 labs(x = "Height (m)", y = "Volume (m³)", title = "Volume vs. Height") + theme_forest())
    })
    mens_upsert("General Formula", vol)
  })
  
  observeEvent(input$calc_smalian, {
    vol <- ((input$smalian_a1 + input$smalian_a2) / 2) * input$smalian_l
    output$smalian_result <- renderText(paste("Smalian Volume =", round(vol, 4), "m³"))
    a2_seq <- seq(0.01, 0.12, by = 0.01)
    output$smalian_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(A2 = a2_seq, V = ((input$smalian_a1 + a2_seq) / 2) * input$smalian_l), aes(x = A2, y = V)) +
                 geom_line(color = "#a3612a", size = 1.1) +
                 geom_point(data = data.frame(A2 = input$smalian_a2, V = vol), color = "#d9a21b", size = 4) +
                 labs(x = "Small End Area (m²)", y = "Volume (m³)", title = "Smalian Volume") + theme_forest())
    })
    mens_upsert("Smalian's Formula", vol)
  })
  
  observeEvent(input$calc_huber, {
    vol <- input$huber_am * input$huber_l
    output$huber_result <- renderText(paste("Huber Volume =", round(vol, 4), "m³"))
    am_seq <- seq(0.01, 0.12, by = 0.01)
    output$huber_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(Am = am_seq, V = am_seq * input$huber_l), aes(x = Am, y = V)) +
                 geom_line(color = "#4f8f74", size = 1.1) +
                 geom_point(data = data.frame(Am = input$huber_am, V = vol), color = "#d9a21b", size = 4) +
                 labs(x = "Mid-length Area (m²)", y = "Volume (m³)", title = "Huber Volume") + theme_forest())
    })
    mens_upsert("Huber's Formula", vol)
  })
  
  observeEvent(input$calc_newton, {
    vol <- ((input$newton_a1 + 4 * input$newton_am + input$newton_a2) / 6) * input$newton_l
    output$newton_result <- renderText(paste("Newton Volume =", round(vol, 4), "m³"))
    am_seq <- seq(0.01, 0.12, by = 0.005)
    output$newton_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(Am = am_seq, V = ((input$newton_a1 + 4 * am_seq + input$newton_a2) / 6) * input$newton_l), aes(x = Am, y = V)) +
                 geom_line(color = "#b8892b", size = 1.1) +
                 geom_point(data = data.frame(Am = input$newton_am, V = vol), color = "#d9a21b", size = 4) +
                 labs(x = "Mid Area (m²)", y = "Volume (m³)", title = "Newton Volume") + theme_forest())
    })
    mens_upsert("Newton's Formula", vol)
  })
  
  observeEvent(input$calc_fq, {
    fq <- input$fq_dh / input$fq_dbh
    output$fq_result <- renderText(paste("Form Quotient (FQ) =", round(fq, 3)))
    dh_seq <- seq(0.5 * input$fq_dbh, 1.5 * input$fq_dbh, by = 0.5)
    output$fq_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(Dh = dh_seq, FQ = dh_seq / input$fq_dbh), aes(x = Dh, y = FQ)) +
                 geom_line(color = "#3d5f7a", size = 1.1) +
                 geom_point(data = data.frame(Dh = input$fq_dh, FQ = fq), color = "#d9a21b", size = 4) +
                 labs(x = "Diameter at Half Height (cm)", y = "Form Quotient", title = "Form Quotient") + theme_forest())
    })
  })
  
  observeEvent(input$calc_qg, {
    vol <- (input$qg_girth^2 * input$qg_length) / 40000
    output$qg_result <- renderText(paste("Quarter Girth Volume =", round(vol, 4), "m³"))
    g_seq <- seq(50, 200, by = 10)
    output$qg_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(G = g_seq, V = (g_seq^2 * input$qg_length) / 40000), aes(x = G, y = V)) +
                 geom_line(color = "#d9a21b", size = 1.1) +
                 geom_point(data = data.frame(G = input$qg_girth, V = vol), color = "#1d4a3a", size = 4) +
                 labs(x = "Girth (cm)", y = "Volume (m³)", title = "Quarter Girth Volume") + theme_forest())
    })
    mens_upsert("Quarter Girth", vol)
  })
  
  observeEvent(input$calc_approx, {
    vol <- (input$approx_dbh^2 * input$approx_height) / 1000
    output$approx_result <- renderText(paste("Approximate Volume =", round(vol, 4), "m³"))
    d_seq <- seq(10, 60, by = 5)
    output$approx_viz <- renderPlotly({
      ggplotly(ggplot(data.frame(D = d_seq, V = (d_seq^2 * input$approx_height) / 1000), aes(x = D, y = V)) +
                 geom_line(color = "#2f6b52", size = 1.1) +
                 geom_point(data = data.frame(D = input$approx_dbh, V = vol), color = "#d9a21b", size = 4) +
                 labs(x = "DBH (cm)", y = "Volume (m³)", title = "Approx. Volume vs. DBH") + theme_forest())
    })
    mens_upsert("Quick Approximation", vol)
  })
  
  output$volume_comparison_table <- renderDT({
    df <- mens_results$volumes
    df$Volume <- round(df$Volume, 4)
    datatable(df, rownames = FALSE, options = list(pageLength = 6, dom = 'tip'))
  })
  
  output$volume_comparison_plot <- renderPlotly({
    df <- mens_results$volumes
    validate(need(nrow(df) > 0, "Volumes appear here as the calculators above run."))
    df <- df[order(df$Volume), ]
    df$Method <- factor(df$Method, levels = df$Method)
    p <- ggplot(df, aes(x = Method, y = Volume, text = paste0(Method, ": ", round(Volume, 4), " m\u00b3"))) +
      geom_col(fill = forest_series[seq_len(nrow(df))], width = 0.6) +
      geom_hline(yintercept = mean(df$Volume), linetype = "dashed", colour = "#d9a21b") +
      coord_flip() + labs(x = NULL, y = "Volume (m\u00b3)", title = "Dashed line: mean of all methods") + theme_forest()
    ggplotly(p, tooltip = "text") %>% config(displaylogo = FALSE)
  })
  
  observeEvent(input$generate_tree_inputs, {
    num <- input$multi_num_trees; formula <- input$multi_formula
    output$multi_tree_inputs <- renderUI({
      inputs <- lapply(1:num, function(i) {
        if (formula == "general") {
          fluidRow(column(12, h5(paste("Tree", i))),
                   column(4, numericInput(paste0("multi_ba_", i), "Basal Area (m²)", 0.07, min = 0, step = 0.01)),
                   column(4, numericInput(paste0("multi_height_", i), "Height (m)", 20, min = 0)),
                   column(4, numericInput(paste0("multi_form_", i), "Form Factor", 0.5, min = 0, max = 1, step = 0.05)))
        } else {
          fluidRow(column(12, h5(paste("Tree", i))),
                   column(6, numericInput(paste0("multi_dbh_", i), "DBH (cm)", 30, min = 0)),
                   column(6, numericInput(paste0("multi_height_", i), "Height (m)", 20, min = 0)))
        }
      })
      do.call(tagList, inputs)
    })
  })
  
  observeEvent(input$calc_multi_trees, {
    num <- input$multi_num_trees; formula <- input$multi_formula
    tree_data <- data.frame(Tree = 1:num, Volume = numeric(num))
    for (i in 1:num) {
      if (formula == "general") {
        tree_data$Volume[i] <- input[[paste0("multi_ba_", i)]] * input[[paste0("multi_height_", i)]] * input[[paste0("multi_form_", i)]]
      } else {
        tree_data$Volume[i] <- (input[[paste0("multi_dbh_", i)]]^2 * input[[paste0("multi_height_", i)]]) / 1000
      }
    }
    output$multi_tree_results <- renderDT(datatable(tree_data))
    output$multi_tree_viz <- renderPlotly({
      ggplotly(ggplot(tree_data, aes(x = Tree, y = Volume)) +
                 geom_bar(stat = "identity", fill = "#1d4a3a") +
                 labs(x = "Tree Number", y = "Volume (m³)", title = "Volume by Tree") + theme_forest())
    })
  })
  
  # ============================================================
  # DENDROFIT ML SERVER
  # ============================================================
  hd_values <- reactiveValues(
    data = NULL, models_fit = list(), ml_models_fit = list(),
    train_data = NULL, test_data = NULL,
    all_trad_results = NULL, all_ml_results = NULL,
    train_trad_results = NULL, test_trad_results = NULL,
    train_ml_results = NULL, test_ml_results = NULL,
    best_trad_model = NULL, best_ml_model = NULL,
    ann_d_min = NULL, ann_d_max = NULL, ann_h_min = NULL, ann_h_max = NULL
  )
  
  observeEvent(input$hd_dataFile, {
    req(input$hd_dataFile)
    tryCatch({
      data <- read.csv(input$hd_dataFile$datapath)
      hd_values$data <- data
      updateSelectInput(session, "hd_diameterCol", choices = colnames(data))
      updateSelectInput(session, "hd_heightCol", choices = colnames(data))
    }, error = function(e) showNotification("Error reading CSV file.", type = "error"))
  })
  
  observeEvent(input$hd_demo, {
    set.seed(2026)
    d <- round(runif(150, 6, 65), 1)
    h <- round(pmax(1.6, 1.37 + 32 * (1 - exp(-0.045 * d))^1.2 + rnorm(150, 0, 1.6)), 1)
    demo <- data.frame(Tree = 1:150, DBH_cm = d, Height_m = h)
    hd_values$data <- demo
    updateSelectInput(session, "hd_diameterCol", choices = colnames(demo), selected = "DBH_cm")
    updateSelectInput(session, "hd_heightCol", choices = colnames(demo), selected = "Height_m")
    showNotification("Loaded 150 demo trees. Pick models and run the H-D analysis.", type = "message")
  })
  
  output$hd_dataSummary <- renderPrint({ req(hd_values$data); summary(hd_values$data) })
  
  output$hd_dataPreview <- renderDT({
    req(hd_values$data)
    datatable(hd_values$data, options = list(pageLength = 5, scrollX = TRUE))
  })
  
  output$hd_scatterPlot <- renderPlotly({
    req(hd_values$data, input$hd_diameterCol, input$hd_heightCol)
    ggplotly(ggplot(hd_values$data, aes_string(x = input$hd_diameterCol, y = input$hd_heightCol)) +
               geom_point(alpha = 0.7, color = "#2f6b52", size = 2) +
               labs(title = "Height vs. Diameter") + theme_forest())
  })
  
  output$hd_totalObsBox <- renderValueBox({
    valueBox(if (!is.null(hd_values$data)) nrow(hd_values$data) else "N/A",
             "Total Observations", icon = icon("list"), color = "green")
  })
  output$hd_bestModelBox <- renderValueBox({
    valueBox(if (!is.null(hd_values$best_trad_model)) hd_values$best_trad_model else "N/A",
             "Best Traditional Model", icon = icon("chart-line"), color = "blue")
  })
  output$hd_bestMLModelBox <- renderValueBox({
    valueBox(if (!is.null(hd_values$best_ml_model)) hd_values$best_ml_model else "N/A",
             "Best ML Model", icon = icon("brain"), color = "purple")
  })
  output$hd_accuracyBox <- renderValueBox({
    val <- "N/A"
    if (!is.null(hd_values$test_trad_results) && nrow(hd_values$test_trad_results) > 0) {
      best_row <- hd_values$test_trad_results[which.min(hd_values$test_trad_results$RMSE), ]
      val <- paste0("R²=", round(best_row$R2, 3))
    }
    valueBox(val, "Best R² (Test)", icon = icon("bullseye"), color = "orange")
  })
  output$hd_quickStats <- renderPrint({
    req(hd_values$data, input$hd_diameterCol, input$hd_heightCol)
    cat("Diameter summary:\n"); print(summary(hd_values$data[[input$hd_diameterCol]]))
    cat("\nHeight summary:\n"); print(summary(hd_values$data[[input$hd_heightCol]]))
  })
  output$hd_quickModelStats <- renderDT({
    req(hd_values$test_trad_results)
    datatable(head(hd_values$test_trad_results[order(hd_values$test_trad_results$RMSE), ], 3),
              options = list(dom = 't'))
  })
  output$hd_quickMLModelStats <- renderDT({
    req(hd_values$test_ml_results)
    datatable(head(hd_values$test_ml_results[order(hd_values$test_ml_results$RMSE), ], 3),
              options = list(dom = 't'))
  })
  
  hd_predict_ml <- function(model, new_data, model_type, smooth = FALSE,
                            d_min = NULL, d_max = NULL, h_min = NULL, h_max = NULL) {
    predictions <- tryCatch({
      if (model_type == "gbm") predict(model, newdata = new_data, n.trees = model$n.trees)
      else if (model_type == "ann") {
        norm_d <- (new_data$Diameter - d_min) / (d_max - d_min)
        raw    <- neuralnet::compute(model, data.frame(Diameter = norm_d))$net.result
        as.vector(raw) * (h_max - h_min) + h_min
      }
      else predict(model, newdata = new_data)
    }, error = function(e) rep(NA, nrow(new_data)))
    if (smooth && !all(is.na(predictions))) {
      smooth_data <- data.frame(Diameter = new_data$Diameter, Predicted = predictions)
      sm <- tryCatch(gam(Predicted ~ s(Diameter, k = min(10, nrow(smooth_data) - 1)), data = smooth_data), error = function(e) NULL)
      if (!is.null(sm)) predictions <- predict(sm, newdata = new_data)
    }
    predictions
  }
  
  hd_fit_traditional <- function(data, diam_col, height_col, selected_models, train_pct = 0.8) {
    df <- data.frame(Diameter = data[[diam_col]], Height = data[[height_col]])
    df <- df[complete.cases(df), ]
    set.seed(42)
    idx <- sample(nrow(df), train_pct * nrow(df))
    train <- df[idx, ]; test <- df[-idx, ]
    
    model_fns <- list(
      naslund1  = function(d) tryCatch(nlsLM(Height ~ 1.37 + (Diameter^2 / (a + b * Diameter)^2), data = d, start = list(a = 1, b = 1)), error = function(e) NULL),
      curtis    = function(d) tryCatch(nlsLM(Height ~ 1.37 + a * (Diameter / (1 + Diameter))^b, data = d, start = list(a = 10, b = 1)), error = function(e) NULL),
      michailoff= function(d) tryCatch(nlsLM(Height ~ 1.37 + a * exp(-b * Diameter^(-1)), data = d, start = list(a = 30, b = 3)), error = function(e) NULL),
      naslund2  = function(d) tryCatch(nlsLM(Height ~ 1.37 + Diameter^2 / (a + b * Diameter)^2, data = d, start = list(a = 0.5, b = 0.1)), error = function(e) NULL),
      power     = function(d) tryCatch(nlsLM(Height ~ 1.37 + a * Diameter^b, data = d, start = list(a = 1, b = 0.5)), error = function(e) NULL),
      naslund3  = function(d) tryCatch(nlsLM(Height ~ 1.37 + Diameter^3 / (a + b * Diameter)^3, data = d, start = list(a = 0.5, b = 0.1)), error = function(e) NULL),
      naslund4  = function(d) tryCatch(nlsLM(Height ~ 1.37 + Diameter^1.5 / (a + b * Diameter)^1.5, data = d, start = list(a = 0.5, b = 0.1)), error = function(e) NULL),
      michail_menten = function(d) tryCatch(nlsLM(Height ~ 1.37 + (a * Diameter) / (b + Diameter), data = d, start = list(a = 30, b = 10)), error = function(e) NULL),
      michail_menten2= function(d) tryCatch(nlsLM(Height ~ 1.37 + a / (1 + b * exp(-Diameter)), data = d, start = list(a = 30, b = 1)), error = function(e) NULL),
      wykoff    = function(d) tryCatch(nlsLM(Height ~ 1.37 + exp(a + b / (Diameter + 1)), data = d, start = list(a = 3, b = -3)), error = function(e) NULL)
    )
    
    results <- data.frame(Model = character(), RMSE = numeric(), MAE = numeric(), R2 = numeric())
    train_res <- results; test_res <- results; all_models <- list()
    for (nm in selected_models) {
      fn <- model_fns[[nm]]; if (is.null(fn)) next
      m <- fn(train); if (is.null(m)) next
      all_models[[nm]] <- m
      for (split in c("train", "test")) {
        d <- if (split == "train") train else test
        pred <- tryCatch(predict(m, newdata = d), error = function(e) rep(NA, nrow(d)))
        if (all(is.na(pred))) next
        rmse_v <- sqrt(mean((d$Height - pred)^2, na.rm = TRUE))
        mae_v <- mean(abs(d$Height - pred), na.rm = TRUE)
        r2_v <- cor(d$Height, pred, use = "complete.obs")^2
        row <- data.frame(Model = nm, RMSE = round(rmse_v, 2), MAE = round(mae_v, 2), R2 = round(r2_v, 2))
        if (split == "train") train_res <- rbind(train_res, row)
        else test_res <- rbind(test_res, row)
        results <- rbind(results, cbind(Split = split, row))
      }
    }
    list(models = all_models, train = train_res, test = test_res, all = results, train_data = train, test_data = test)
  }
  
  hd_fit_ml <- function(train_data, test_data, selected_models, ntree, mtry, n_gbm, svm_kernel, svm_cost, cp_val,
                        ann_hidden1 = 8, ann_hidden2 = 4, ann_thresh = 0.05) {
    results_train <- data.frame(Model = character(), RMSE = numeric(), MAE = numeric(), R2 = numeric())
    results_test  <- data.frame(Model = character(), RMSE = numeric(), MAE = numeric(), R2 = numeric())
    models <- list()
    
    # Normalise data once for ANN
    d_min <- min(train_data$Diameter); d_max <- max(train_data$Diameter)
    h_min <- min(train_data$Height);   h_max <- max(train_data$Height)
    norm  <- function(x, mn, mx) (x - mn) / (mx - mn)
    denorm<- function(x, mn, mx) x * (mx - mn) + mn
    train_nn <- data.frame(Diameter = norm(train_data$Diameter, d_min, d_max),
                           Height   = norm(train_data$Height,   h_min, h_max))
    test_nn  <- data.frame(Diameter = norm(test_data$Diameter,  d_min, d_max),
                           Height   = norm(test_data$Height,    h_min, h_max))
    
    for (nm in selected_models) {
      m <- tryCatch({
        if (nm == "dt") rpart(Height ~ Diameter, data = train_data, control = rpart.control(cp = cp_val))
        else if (nm == "rf") randomForest(Height ~ Diameter, data = train_data, ntree = ntree, mtry = mtry)
        else if (nm == "svm") svm(Height ~ Diameter, data = train_data, kernel = svm_kernel, cost = svm_cost, scale = TRUE)
        else if (nm == "gbm") gbm(Height ~ Diameter, data = train_data, distribution = "gaussian",
                                  n.trees = n_gbm, interaction.depth = 1, shrinkage = 0.1, verbose = FALSE)
        else if (nm == "ann") {
          hidden_layers <- if (ann_hidden2 > 0) c(ann_hidden1, ann_hidden2) else ann_hidden1
          neuralnet(Height ~ Diameter, data = train_nn, hidden = hidden_layers,
                    threshold = ann_thresh, stepmax = 1e6, linear.output = TRUE)
        }
      }, error = function(e) NULL)
      if (is.null(m)) next
      models[[nm]] <- m
      for (split in c("train", "test")) {
        d     <- if (split == "train") train_data else test_data
        d_nn  <- if (split == "train") train_nn   else test_nn
        pred <- tryCatch({
          if (nm == "gbm") predict(m, newdata = d, n.trees = n_gbm)
          else if (nm == "ann") {
            raw <- neuralnet::compute(m, d_nn["Diameter"])$net.result
            denorm(as.vector(raw), h_min, h_max)
          }
          else predict(m, newdata = d)
        }, error = function(e) rep(NA, nrow(d)))
        if (all(is.na(pred))) next
        row <- data.frame(Model = nm,
                          RMSE = round(sqrt(mean((d$Height - pred)^2, na.rm = TRUE)), 2),
                          MAE  = round(mean(abs(d$Height - pred), na.rm = TRUE), 2),
                          R2   = round(cor(d$Height, pred, use = "complete.obs")^2, 2))
        if (split == "train") results_train <- rbind(results_train, row)
        else results_test <- rbind(results_test, row)
      }
    }
    list(models = models, train = results_train, test = results_test)
  }
  
  observeEvent(input$hd_runAnalysis, {
    req(hd_values$data, input$hd_diameterCol, input$hd_heightCol)
    if (length(input$hd_models) == 0 && length(input$hd_ml_models) == 0) {
      showNotification("Please select at least one model.", type = "warning"); return()
    }
    withProgress(message = "Fitting H-D models...", value = 0, {
      train_pct <- if (input$hd_doCrossValidation) input$hd_trainPercent / 100 else 0.8
      
      if (length(input$hd_models) > 0) {
        incProgress(0.4, detail = "Fitting traditional models...")
        res <- hd_fit_traditional(hd_values$data, input$hd_diameterCol, input$hd_heightCol,
                                  input$hd_models, train_pct)
        hd_values$models_fit <- res$models
        hd_values$train_trad_results <- res$train
        hd_values$test_trad_results  <- res$test
        hd_values$all_trad_results   <- res$all
        hd_values$train_data <- res$train_data
        hd_values$test_data  <- res$test_data
        if (nrow(res$test) > 0) hd_values$best_trad_model <- res$test[which.min(res$test$RMSE), "Model"]
      }
      
      if (length(input$hd_ml_models) > 0 && !is.null(hd_values$train_data)) {
        incProgress(0.4, detail = "Fitting ML models...")
        ml_res <- hd_fit_ml(hd_values$train_data, hd_values$test_data, input$hd_ml_models,
                            input$hd_ntree, input$hd_rf_mtry, input$hd_n_trees_gbm,
                            input$hd_svm_kernel, input$hd_svm_cost, input$hd_cp,
                            ann_hidden1 = input$hd_ann_hidden1,
                            ann_hidden2 = input$hd_ann_hidden2,
                            ann_thresh  = input$hd_ann_thresh)
        hd_values$ml_models_fit   <- ml_res$models
        hd_values$train_ml_results <- ml_res$train
        hd_values$test_ml_results  <- ml_res$test
        hd_values$all_ml_results   <- ml_res$test
        if (nrow(ml_res$test) > 0) hd_values$best_ml_model <- ml_res$test[which.min(ml_res$test$RMSE), "Model"]
        # Store ANN normalization bounds for prediction
        hd_values$ann_d_min <- min(hd_values$train_data$Diameter)
        hd_values$ann_d_max <- max(hd_values$train_data$Diameter)
        hd_values$ann_h_min <- min(hd_values$train_data$Height)
        hd_values$ann_h_max <- max(hd_values$train_data$Height)
      }
      incProgress(0.2, detail = "Done.")
    })
    showNotification("H-D Analysis complete!", type = "message")
  })
  
  output$hd_modelComparison  <- renderDT({ req(hd_values$test_trad_results); datatable(hd_values$test_trad_results) %>% formatRound(c("RMSE","MAE","R2"), 2) })
  output$hd_mlModelComparison<- renderDT({ req(hd_values$all_ml_results);   datatable(hd_values$all_ml_results)   %>% formatRound(c("RMSE","MAE","R2"), 2) })
  output$hd_modelParams <- renderPrint({
    req(hd_values$models_fit)
    for (nm in names(hd_values$models_fit)) { cat("\nModel:", nm, "\n"); print(summary(hd_values$models_fit[[nm]])) }
  })
  output$hd_trainResults   <- renderDT({ req(hd_values$train_trad_results); datatable(hd_values$train_trad_results) %>% formatRound(c("RMSE","MAE","R2"), 2) })
  output$hd_testResults    <- renderDT({ req(hd_values$test_trad_results);  datatable(hd_values$test_trad_results)  %>% formatRound(c("RMSE","MAE","R2"), 2) })
  output$hd_mlTrainResults <- renderDT({ req(hd_values$train_ml_results);   datatable(hd_values$train_ml_results)   %>% formatRound(c("RMSE","MAE","R2"), 2) })
  output$hd_mlTestResults  <- renderDT({ req(hd_values$test_ml_results);    datatable(hd_values$test_ml_results)    %>% formatRound(c("RMSE","MAE","R2"), 2) })
  output$hd_combinedResults <- renderDT({
    req(hd_values$test_trad_results, hd_values$test_ml_results)
    combined <- rbind(cbind(Type = "Traditional", hd_values$test_trad_results),
                      cbind(Type = "ML", hd_values$test_ml_results))
    datatable(combined) %>% formatRound(c("RMSE","MAE","R2"), 2)
  })
  output$hd_dtPlot    <- renderPlot({ req(hd_values$ml_models_fit$dt); rpart.plot(hd_values$ml_models_fit$dt, main = "Decision Tree") })
  output$hd_dtSummary <- renderPrint({ req(hd_values$ml_models_fit$dt); print(summary(hd_values$ml_models_fit$dt)) })
  output$hd_rfPlot    <- renderPlot({ req(hd_values$ml_models_fit$rf); plot(hd_values$ml_models_fit$rf, main = "RF Error vs. Trees") })
  output$hd_rfSummary <- renderPrint({ req(hd_values$ml_models_fit$rf); print(hd_values$ml_models_fit$rf) })
  output$hd_svmSummary <- renderPrint({ req(hd_values$ml_models_fit$svm); print(summary(hd_values$ml_models_fit$svm)) })
  output$hd_gbmPlot   <- renderPlot({ req(hd_values$ml_models_fit$gbm); plot(hd_values$ml_models_fit$gbm, main = "GBM Performance") })
  output$hd_gbmSummary <- renderPrint({ req(hd_values$ml_models_fit$gbm); print(summary(hd_values$ml_models_fit$gbm)) })
  
  output$hd_annPlot <- renderPlot({
    req(hd_values$ml_models_fit$ann)
    tryCatch(plot(hd_values$ml_models_fit$ann, main = "Neural Network Architecture"), error = function(e) {
      plot.new(); text(0.5, 0.5, "ANN structure plot unavailable.\nSee summary for weights.", cex = 1.2)
    })
  })
  output$hd_annSummary <- renderPrint({
    req(hd_values$ml_models_fit$ann)
    ann <- hd_values$ml_models_fit$ann
    cat("=== Neural Network (ANN) Summary ===\n")
    cat("Hidden layers : ", paste(ann$hidden, collapse = " -> "), "\n")
    cat("Algorithm     : ", ann$call$algorithm %||% "rprop+", "\n")
    cat("Threshold     : ", ann$call$threshold %||% 0.05, "\n")
    cat("Steps         : ", ann$result.matrix["steps", 1], "\n")
    cat("Error         : ", round(ann$result.matrix["error", 1], 6), "\n")
    cat("\nWeight matrix (1st result):\n")
    print(ann$weights[[1]])
  })
  
  output$hd_modelPlot <- renderPlotly({
    req(hd_values$data, hd_values$models_fit, input$hd_diameterCol, input$hd_heightCol)
    pd <- data.frame(Diameter = hd_values$data[[input$hd_diameterCol]], Height = hd_values$data[[input$hd_heightCol]])
    pd <- pd[complete.cases(pd), ]
    p <- plot_ly() %>% add_trace(x = pd$Diameter, y = pd$Height, type = 'scatter', mode = 'markers',
                                 name = 'Data', marker = list(color = 'rgba(111,138,134,0.55)'))
    cols <- forest_series
    for (i in seq_along(names(hd_values$models_fit))) {
      nm <- names(hd_values$models_fit)[i]; m <- hd_values$models_fit[[nm]]
      nd <- data.frame(Diameter = seq(min(pd$Diameter), max(pd$Diameter), length.out = 200))
      preds <- tryCatch(predict(m, newdata = nd), error = function(e) rep(NA, 200))
      p <- add_trace(p, x = nd$Diameter, y = preds, type = 'scatter', mode = 'lines', name = nm,
                     line = list(color = cols[(i - 1) %% length(cols) + 1], width = 2))
    }
    layout(p, title = "Traditional Models Fit", xaxis = list(title = "Diameter"), yaxis = list(title = "Height"))
  })
  
  output$hd_mlModelPlot <- renderPlotly({
    req(hd_values$data, hd_values$ml_models_fit, input$hd_diameterCol, input$hd_heightCol)
    pd <- data.frame(Diameter = hd_values$data[[input$hd_diameterCol]], Height = hd_values$data[[input$hd_heightCol]])
    pd <- pd[complete.cases(pd), ]
    p <- plot_ly() %>% add_trace(x = pd$Diameter, y = pd$Height, type = 'scatter', mode = 'markers',
                                 name = 'Data', marker = list(color = 'rgba(111,138,134,0.55)'))
    cols <- forest_series
    for (i in seq_along(names(hd_values$ml_models_fit))) {
      nm <- names(hd_values$ml_models_fit)[i]; m <- hd_values$ml_models_fit[[nm]]
      nd <- data.frame(Diameter = seq(min(pd$Diameter), max(pd$Diameter), length.out = 200))
      preds <- hd_predict_ml(m, nd, nm, smooth = TRUE,
                             d_min = hd_values$ann_d_min, d_max = hd_values$ann_d_max,
                             h_min = hd_values$ann_h_min, h_max = hd_values$ann_h_max)
      p <- add_trace(p, x = nd$Diameter, y = preds, type = 'scatter', mode = 'lines', name = nm,
                     line = list(color = cols[(i - 1) %% length(cols) + 1], width = 2))
    }
    layout(p, title = "ML Models Fit", xaxis = list(title = "Diameter"), yaxis = list(title = "Height"))
  })
  
  output$hd_dashboardPlot <- renderPlotly({
    req(hd_values$data, input$hd_diameterCol, input$hd_heightCol)
    pd <- data.frame(Diameter = hd_values$data[[input$hd_diameterCol]], Height = hd_values$data[[input$hd_heightCol]])
    pd <- pd[complete.cases(pd), ]
    p <- plot_ly() %>% add_trace(x = pd$Diameter, y = pd$Height, type = 'scatter', mode = 'markers',
                                 name = 'Data', marker = list(color = 'rgba(111,138,134,0.55)'))
    if (!is.null(hd_values$best_trad_model) && hd_values$best_trad_model %in% names(hd_values$models_fit)) {
      m <- hd_values$models_fit[[hd_values$best_trad_model]]
      nd <- data.frame(Diameter = seq(min(pd$Diameter), max(pd$Diameter), length.out = 200))
      preds <- tryCatch(predict(m, newdata = nd), error = function(e) rep(NA, 200))
      p <- add_trace(p, x = nd$Diameter, y = preds, type = 'scatter', mode = 'lines',
                     name = paste("Best Trad:", hd_values$best_trad_model),
                     line = list(color = 'rgba(29,74,58,1)', width = 3))
    }
    if (!is.null(hd_values$best_ml_model) && hd_values$best_ml_model %in% names(hd_values$ml_models_fit)) {
      m <- hd_values$ml_models_fit[[hd_values$best_ml_model]]
      nd <- data.frame(Diameter = seq(min(pd$Diameter), max(pd$Diameter), length.out = 200))
      preds <- hd_predict_ml(m, nd, hd_values$best_ml_model, smooth = TRUE,
                             d_min = hd_values$ann_d_min, d_max = hd_values$ann_d_max,
                             h_min = hd_values$ann_h_min, h_max = hd_values$ann_h_max)
      p <- add_trace(p, x = nd$Diameter, y = preds, type = 'scatter', mode = 'lines',
                     name = paste("Best ML:", hd_values$best_ml_model),
                     line = list(color = 'rgba(217,162,27,1)', width = 3))
    }
    layout(p, title = "Tree Height-Diameter Relationship", xaxis = list(title = "Diameter"), yaxis = list(title = "Height"))
  })
  
  output$hd_validationPlot <- renderPlotly({
    req(hd_values$test_data)
    pd <- hd_values$test_data
    p <- plot_ly() %>% add_trace(x = pd$Diameter, y = pd$Height, type = 'scatter', mode = 'markers',
                                 name = 'Actual', marker = list(color = 'rgba(111,138,134,0.55)'))
    if (!is.null(hd_values$best_trad_model) && hd_values$best_trad_model %in% names(hd_values$models_fit)) {
      m <- hd_values$models_fit[[hd_values$best_trad_model]]
      nd <- data.frame(Diameter = sort(pd$Diameter))
      preds <- tryCatch(predict(m, newdata = nd), error = function(e) rep(NA, nrow(nd)))
      p <- add_trace(p, x = nd$Diameter, y = preds, type = 'scatter', mode = 'markers',
                     name = paste("Best Trad:", hd_values$best_trad_model),
                     marker = list(color = 'rgba(29,74,58,0.7)', symbol = 'x'))
    }
    if (!is.null(hd_values$best_ml_model) && hd_values$best_ml_model %in% names(hd_values$ml_models_fit)) {
      m <- hd_values$ml_models_fit[[hd_values$best_ml_model]]
      nd <- data.frame(Diameter = seq(min(pd$Diameter), max(pd$Diameter), length.out = 200))
      preds <- hd_predict_ml(m, nd, hd_values$best_ml_model, smooth = TRUE)
      p <- add_trace(p, x = nd$Diameter, y = preds, type = 'scatter', mode = 'lines',
                     name = paste("Best ML:", hd_values$best_ml_model),
                     line = list(color = 'rgba(217,162,27,1)', width = 2))
    }
    layout(p, title = "Validation: Actual vs. Predicted", xaxis = list(title = "Diameter"), yaxis = list(title = "Height"))
  })
  
  # ============================================================
  # DENDROVOL ML SERVER
  # ============================================================
  dv_demo_data <- data.frame(
    Height   = c(6.70, 10.70, 16.70, 22.30, 26.20, 29.90, 12.5, 18.2, 21.0, 24.5),
    Diameter = c(12.20, 14.70, 19.30, 24.10, 28.70, 33.50, 16.8, 22.3, 25.1, 30.0),
    Volume   = c(0.02, 0.05, 0.15, 0.31, 0.51, 0.80, 0.08, 0.20, 0.35, 0.55)
  )
  
  dv_dataset        <- reactiveVal(dv_demo_data)
  dv_model_results  <- reactiveVal(NULL)
  dv_fitted_models  <- reactiveVal(NULL)
  dv_ml_results_rv  <- reactiveVal(NULL)
  dv_ml_models_rv   <- reactiveVal(NULL)
  dv_predictions    <- reactiveVal(NULL)
  
  `%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !is.na(a)) a else b
  
  observe({
    df <- dv_dataset()
    if (!is.null(df)) {
      cols <- names(df)
      updateSelectInput(session, "dv_diameter_var", choices = cols,
                        selected = cols[grep("diameter|dbh", tolower(cols))[1] %||% 1])
      updateSelectInput(session, "dv_height_var",   choices = cols,
                        selected = cols[grep("height", tolower(cols))[1] %||% 1])
      updateSelectInput(session, "dv_volume_var",   choices = cols,
                        selected = cols[grep("volume|vol", tolower(cols))[1] %||% 1])
    }
  })
  
  observeEvent(input$dv_datafile, {
    req(input$dv_datafile)
    df <- try(read.csv(input$dv_datafile$datapath), silent = TRUE)
    if (!inherits(df, "try-error")) {
      dv_dataset(df)
      dv_model_results(NULL); dv_fitted_models(NULL)
      dv_ml_results_rv(NULL); dv_ml_models_rv(NULL); dv_predictions(NULL)
    } else showNotification("Error reading CSV file.", type = "error")
  })
  
  output$dv_data_preview <- renderDT({ req(dv_dataset()); datatable(dv_dataset(), options = list(pageLength = 10, scrollX = TRUE)) })
  
  dv_fit_allometric <- function(train_data, test_data, selected_models) {
    results <- data.frame(Model = character(), RMSE = numeric(), MAE = numeric(), R2 = numeric())
    models <- list()
    fns <- list(
      Linear.D1       = function(d) lm(Volume ~ Diameter, data = d),
      Logarithmic.D2  = function(d) lm(Volume ~ log(Diameter), data = d),
      Inverse.D3      = function(d) lm(Volume ~ I(1 / Diameter), data = d),
      Quadratic.D4    = function(d) lm(Volume ~ Diameter + I(Diameter^2), data = d),
      Cubic.D5        = function(d) lm(Volume ~ Diameter + I(Diameter^2) + I(Diameter^3), data = d),
      Power.D7        = function(d) tryCatch(nls2(Volume ~ a * Diameter^b, start = list(a = 0.1, b = 2), algorithm = "port", data = d), error = function(e) NULL),
      Exponential.D8  = function(d) tryCatch(nls2(Volume ~ a * exp(b * Diameter), start = list(a = 0.1, b = 0.1), algorithm = "port", data = d), error = function(e) NULL),
      Linear.HD1      = function(d) lm(Volume ~ Height + Diameter, data = d),
      Power.HD7       = function(d) tryCatch(nls2(Volume ~ a * (Height * Diameter)^b, start = list(a = 0.1, b = 1), algorithm = "port", data = d), error = function(e) NULL)
    )
    for (nm in selected_models) {
      fn <- fns[[nm]]; if (is.null(fn)) next
      m <- tryCatch(fn(train_data), error = function(e) NULL); if (is.null(m)) next
      pred <- tryCatch(predict(m, newdata = test_data), error = function(e) rep(NA, nrow(test_data)))
      if (all(is.na(pred))) next
      results <- rbind(results, data.frame(Model = nm,
                                           RMSE = round(sqrt(mean((test_data$Volume - pred)^2, na.rm = TRUE)), 2),
                                           MAE  = round(mean(abs(test_data$Volume - pred), na.rm = TRUE), 2),
                                           R2   = round(cor(test_data$Volume, pred, use = "complete.obs")^2, 2)))
      models[[nm]] <- m
    }
    list(results = results, models = models)
  }
  
  dv_fit_ml <- function(train_data, test_data, selected_models,
                        ann_hidden1 = 8, ann_hidden2 = 4, ann_thresh = 0.05) {
    results <- data.frame(Model = character(), RMSE = numeric(), MAE = numeric(), R2 = numeric())
    models <- list(); importance_data <- data.frame()
    
    # Normalise for ANN
    d_min <- min(train_data$Diameter); d_max <- max(train_data$Diameter)
    h_min <- min(train_data$Height);   h_max <- max(train_data$Height)
    v_min <- min(train_data$Volume);   v_max <- max(train_data$Volume)
    norm  <- function(x, mn, mx) (x - mn) / (mx - mn)
    denorm<- function(x, mn, mx) x * (mx - mn) + mn
    train_nn <- data.frame(
      Diameter = norm(train_data$Diameter, d_min, d_max),
      Height   = norm(train_data$Height,   h_min, h_max),
      Volume   = norm(train_data$Volume,   v_min, v_max)
    )
    test_nn <- data.frame(
      Diameter = norm(test_data$Diameter, d_min, d_max),
      Height   = norm(test_data$Height,   h_min, h_max)
    )
    
    for (nm in selected_models) {
      m <- tryCatch({
        if (nm == "DT") rpart(Volume ~ Diameter + Height, data = train_data)
        else if (nm == "RF") randomForest(Volume ~ Diameter + Height, data = train_data, ntree = 100, importance = TRUE)
        else if (nm == "SVM") svm(Volume ~ Diameter + Height, data = train_data, scale = TRUE)
        else if (nm == "GB") gbm(Volume ~ Diameter + Height, data = train_data, distribution = "gaussian",
                                 n.trees = 100, interaction.depth = 3, shrinkage = 0.1, verbose = FALSE)
        else if (nm == "ANN") {
          hidden_layers <- if (ann_hidden2 > 0) c(ann_hidden1, ann_hidden2) else ann_hidden1
          neuralnet(Volume ~ Diameter + Height, data = train_nn,
                    hidden = hidden_layers, threshold = ann_thresh,
                    stepmax = 1e6, linear.output = TRUE)
        }
      }, error = function(e) NULL)
      if (is.null(m)) next
      
      pred <- tryCatch({
        if (nm == "GB") predict(m, newdata = test_data, n.trees = 100)
        else if (nm == "ANN") {
          raw <- neuralnet::compute(m, test_nn[, c("Diameter","Height")])$net.result
          denorm(as.vector(raw), v_min, v_max)
        }
        else predict(m, newdata = test_data)
      }, error = function(e) rep(NA, nrow(test_data)))
      if (all(is.na(pred))) next
      
      results <- rbind(results, data.frame(Model = nm,
                                           RMSE = round(sqrt(mean((test_data$Volume - pred)^2, na.rm = TRUE)), 2),
                                           MAE  = round(mean(abs(test_data$Volume - pred), na.rm = TRUE), 2),
                                           R2   = round(cor(test_data$Volume, pred, use = "complete.obs")^2, 2)))
      models[[nm]] <- m
      # store ANN norm params inside model object for later prediction use
      if (nm == "ANN") attr(models[[nm]], "norm_params") <- list(
        d_min=d_min, d_max=d_max, h_min=h_min, h_max=h_max, v_min=v_min, v_max=v_max)
      if (nm == "RF") {
        imp <- importance(m, type = 1)
        importance_data <- rbind(importance_data, data.frame(Variable = rownames(imp), Importance = imp[, 1], Model = "RF"))
      }
    }
    list(results = results, models = models, importance = importance_data)
  }
  
  observeEvent(input$dv_run, {
    req(dv_dataset(), input$dv_diameter_var, input$dv_height_var, input$dv_volume_var)
    withProgress(message = "Fitting volume models...", value = 0, {
      data <- dv_dataset()
      ds <- data.frame(Diameter = data[[input$dv_diameter_var]],
                       Height   = data[[input$dv_height_var]],
                       Volume   = data[[input$dv_volume_var]])
      ds <- ds[complete.cases(ds) & apply(ds, 1, function(x) all(is.finite(x))), ]
      if (nrow(ds) < 6) { showNotification("Insufficient data.", type = "error"); return() }
      splits <- list("50:50" = 0.5, "70:30" = 0.7, "80:20" = 0.8)
      train_r <- splits[[input$dv_split]]
      set.seed(123)
      idx <- sample(nrow(ds), train_r * nrow(ds))
      train <- ds[idx, ]; test <- ds[-idx, ]
      if (nrow(test) < 2) test <- ds  # fallback
      
      incProgress(0.4, detail = "Allometric models...")
      if (length(input$dv_allometric_models) > 0) {
        res <- dv_fit_allometric(train, test, input$dv_allometric_models)
        dv_model_results(res$results); dv_fitted_models(res$models)
      }
      incProgress(0.4, detail = "ML models...")
      if (length(input$dv_ml_models) > 0) {
        ml_res <- dv_fit_ml(train, test, input$dv_ml_models,
                            ann_hidden1 = input$dv_ann_hidden1,
                            ann_hidden2 = input$dv_ann_hidden2,
                            ann_thresh  = input$dv_ann_thresh)
        dv_ml_results_rv(ml_res); dv_ml_models_rv(ml_res$models)
      }
      incProgress(0.2)
    })
    showNotification("Volume analysis complete!", type = "message")
  })
  
  output$dv_best_models <- renderPrint({
    combined <- rbind(
      if (!is.null(dv_model_results())) dv_model_results() else data.frame(),
      if (!is.null(dv_ml_results_rv())) dv_ml_results_rv()$results else data.frame()
    )
    if (nrow(combined) == 0) { cat("No models fitted yet. Please run analysis."); return() }
    combined[, c("RMSE","MAE","R2")] <- lapply(combined[, c("RMSE","MAE","R2")], round, 2)
    cat("Top Performing Models (by RMSE):\n")
    print(head(combined[order(combined$RMSE), ], 5), row.names = FALSE)
  })
  
  output$dv_results_plot <- renderPlot({
    combined <- rbind(
      if (!is.null(dv_model_results())) dv_model_results() else data.frame(),
      if (!is.null(dv_ml_results_rv())) dv_ml_results_rv()$results else data.frame()
    )
    if (nrow(combined) == 0) { plot(1, type="n", main="No models fitted"); return() }
    combined_long <- pivot_longer(combined, cols = c(RMSE, MAE, R2), names_to = "Metric", values_to = "Value")
    ggplot(combined_long, aes(x = Model, y = Value, fill = Metric)) +
      geom_bar(stat = "identity", position = "dodge") +
      facet_wrap(~Metric, scales = "free_y") +
      scale_fill_manual(values = c(MAE = "#6f8a86", R2 = "#d9a21b", RMSE = "#1d4a3a"), guide = "none") +
      theme_forest() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = "Model Performance")
  })
  
  output$dv_scatter_plot <- renderPlot({
    allom <- dv_fitted_models(); ml_m <- dv_ml_models_rv()
    if (is.null(allom) && is.null(ml_m)) { plot(1, type="n", main="No models"); return() }
    data <- dv_dataset()
    ds <- data.frame(Diameter = data[[input$dv_diameter_var]],
                     Height   = data[[input$dv_height_var]],
                     Volume   = data[[input$dv_volume_var]])
    ds <- ds[complete.cases(ds) & apply(ds, 1, function(x) all(is.finite(x))), ]
    set.seed(123); train_r <- list("50:50"=0.5,"70:30"=0.7,"80:20"=0.8)[[input$dv_split]]
    test <- ds[-sample(nrow(ds), train_r * nrow(ds)), ]
    if (nrow(test) < 2) test <- ds
    pd <- data.frame(Observed = test$Volume)
    if (!is.null(allom)) {
      for (nm in names(allom)) {
        pred <- tryCatch(predict(allom[[nm]], newdata = test), error = function(e) rep(NA, nrow(test)))
        pd[[nm]] <- pred
      }
    }
    if (!is.null(ml_m)) {
      for (nm in names(ml_m)) {
        pred <- tryCatch({
          if (nm == "GB") predict(ml_m[[nm]], newdata = test, n.trees = 100)
          else if (nm == "ANN") {
            np <- attr(ml_m[[nm]], "norm_params")
            nd_norm <- data.frame(
              Diameter = (test$Diameter - np$d_min) / (np$d_max - np$d_min),
              Height   = (test$Height   - np$h_min) / (np$h_max - np$h_min)
            )
            raw <- neuralnet::compute(ml_m[[nm]], nd_norm)$net.result
            as.vector(raw) * (np$v_max - np$v_min) + np$v_min
          }
          else predict(ml_m[[nm]], newdata = test)
        }, error = function(e) rep(NA, nrow(test)))
        pd[[nm]] <- pred
      }
    }
    pd_long <- pivot_longer(pd, cols = -Observed, names_to = "Model", values_to = "Predicted")
    ggplot(pd_long, aes(x = Observed, y = Predicted, color = Model)) +
      geom_point() + geom_abline(linetype = "dashed") + theme_forest() +
      labs(title = "Predicted vs. Observed Volume")
  })
  
  output$dv_allometric_model_summaries <- renderPrint({
    m <- dv_fitted_models(); if (is.null(m)) { cat("No models fitted."); return() }
    for (nm in names(m)) { cat("\nModel:", nm, "\n"); print(summary(m[[nm]])) }
  })
  
  output$dv_ml_model_info <- renderPrint({
    r <- dv_ml_results_rv(); if (is.null(r)) { cat("No ML models."); return() }
    res <- r$results
    res[, c("RMSE","MAE","R2")] <- lapply(res[, c("RMSE","MAE","R2")], round, 2)
    cat("ML Model Performance:\n"); print(res, row.names = FALSE)
  })
  
  output$dv_ml_importance_plot <- renderPlot({
    r <- dv_ml_results_rv()
    if (is.null(r) || is.null(r$importance) || nrow(r$importance) == 0) {
      plot(1, type="n", main="No importance data"); return()
    }
    ggplot(r$importance, aes(x = reorder(Variable, Importance), y = Importance, fill = Model)) +
      geom_bar(stat = "identity", position = "dodge") + coord_flip() + theme_forest() +
      labs(title = "Variable Importance", x = "Variable", y = "Importance")
  })
  
  observeEvent(input$dv_predict, {
    req(input$dv_pred_diameter, input$dv_pred_height)
    pd_data <- data.frame(Diameter = input$dv_pred_diameter, Height = input$dv_pred_height)
    allom <- dv_fitted_models(); ml_m <- dv_ml_models_rv()
    pred_results <- data.frame(Model = character(), Predicted_Volume = numeric())
    if (!is.null(allom)) {
      for (nm in names(allom)) {
        pred <- tryCatch(predict(allom[[nm]], newdata = pd_data), error = function(e) NA)
        pred_results <- rbind(pred_results, data.frame(Model = nm, Predicted_Volume = pred))
      }
    }
    if (!is.null(ml_m)) {
      for (nm in names(ml_m)) {
        pred <- tryCatch({
          if (nm == "GB") predict(ml_m[[nm]], newdata = pd_data, n.trees = 100)
          else if (nm == "ANN") {
            np <- attr(ml_m[[nm]], "norm_params")
            nd_norm <- data.frame(
              Diameter = (pd_data$Diameter - np$d_min) / (np$d_max - np$d_min),
              Height   = (pd_data$Height   - np$h_min) / (np$h_max - np$h_min)
            )
            raw <- neuralnet::compute(ml_m[[nm]], nd_norm)$net.result
            as.vector(raw) * (np$v_max - np$v_min) + np$v_min
          }
          else predict(ml_m[[nm]], newdata = pd_data)
        }, error = function(e) NA)
        pred_results <- rbind(pred_results, data.frame(Model = nm, Predicted_Volume = round(pred, 4)))
      }
    }
    dv_predictions(pred_results)
  })
  
  output$dv_prediction_results <- renderPrint({
    pr <- dv_predictions()
    if (is.null(pr)) { cat("No predictions yet. Click 'Predict Volume'."); return() }
    cat("Predicted Volume for Diameter =", input$dv_pred_diameter, "| Height =", input$dv_pred_height, "\n")
    print(pr, row.names = FALSE)
  })
  
  output$dv_download_results <- downloadHandler(
    filename = function() paste0("DendroVol_Results_", Sys.Date(), ".csv"),
    content = function(file) {
      combined <- rbind(
        if (!is.null(dv_model_results())) dv_model_results() else data.frame(),
        if (!is.null(dv_ml_results_rv())) dv_ml_results_rv()$results else data.frame()
      )
      if (nrow(combined) == 0) combined <- data.frame(Message = "No results")
      write.csv(combined, file, row.names = FALSE)
    }
  )
  
  # ============================================================
  # TREECARBONPRO SERVER
  # ============================================================
  tc_values <- reactiveValues(results_df = NULL)
  
  observeEvent(input$tc_calc, {
    req(input$tc_dbh, input$tc_height, input$tc_density, input$tc_volume,
        input$tc_bef, input$tc_rs_ratio, input$tc_carbon_frac)
    
    dbh <- input$tc_dbh; height <- input$tc_height; density <- input$tc_density
    volume <- input$tc_volume; bef <- input$tc_bef; rs <- input$tc_rs_ratio; cf <- input$tc_carbon_frac
    
    agb_allometric <- 0.0509 * density * dbh^2 * height
    agb_bef        <- volume * density * bef
    agb_avg        <- mean(c(agb_allometric, agb_bef))
    bgb            <- agb_avg * rs
    total_biomass  <- agb_avg + bgb
    carbon_stock   <- total_biomass * cf
    
    tc_values$results_df <- data.frame(
      Metric = c("AGB (Allometric - Chave 2005)", "AGB (BEF-based - IPCC 2006)",
                 "Average AGB", "BGB (Mokany et al. 2006)", "Total Biomass", "Estimated Carbon Stock"),
      Value  = c(agb_allometric, agb_bef, agb_avg, bgb, total_biomass, carbon_stock),
      Unit   = c("kg", "kg", "kg", "kg", "kg", "kg C")
    )
    
    output$tc_results <- renderPrint({
      cat("\nAGB (Allometric - Chave 2005):", round(agb_allometric, 2), "kg")
      cat("\nAGB (BEF-based - IPCC 2006):  ", round(agb_bef, 2), "kg")
      cat("\nAverage AGB:                  ", round(agb_avg, 2), "kg")
      cat("\nBGB (Mokany et al. 2006):     ", round(bgb, 2), "kg")
      cat("\nTotal Biomass:                ", round(total_biomass, 2), "kg")
      cat("\nEstimated Carbon Stock:        ", round(carbon_stock, 2), "kg C")
    })
    showNotification("Carbon stock estimated.", type = "message")
    updateTabItems(session, "main_sidebar", "carbon_results")
  })
  
  tc_val <- function(i) if (is.null(tc_values$results_df)) NA else tc_values$results_df$Value[i]
  tc_fmt <- function(x, unit) if (is.na(x)) "\u2014" else HTML(paste0(formatC(x, format = "f", digits = 1, big.mark = ","), "<small> ", unit, "</small>"))
  
  output$tc_vb_agb <- renderValueBox(valueBox(tc_fmt(tc_val(3), "kg"), "Above-ground biomass (mean)", icon = icon("tree"), color = "green"))
  output$tc_vb_bgb <- renderValueBox(valueBox(tc_fmt(tc_val(4), "kg"), "Below-ground biomass", icon = icon("seedling"), color = "olive"))
  output$tc_vb_total <- renderValueBox(valueBox(tc_fmt(tc_val(5), "kg"), "Total biomass", icon = icon("weight-hanging"), color = "teal"))
  output$tc_vb_carbon <- renderValueBox({
    cs <- tc_val(6)
    sub <- if (is.na(cs)) "Carbon stock" else paste0("Carbon stock, ", formatC(cs * 44 / 12, format = "f", digits = 1, big.mark = ","), " kg CO\u2082e")
    valueBox(tc_fmt(cs, "kg C"), sub, icon = icon("leaf"), color = "yellow")
  })
  
  output$tc_bar <- renderPlotly({
    df <- tc_values$results_df
    validate(need(!is.null(df), "Press Estimate carbon stock in the sidebar to see the breakdown."))
    df$Metric <- factor(df$Metric, levels = rev(df$Metric))
    fills <- c("#8fb39a", "#8fb39a", "#1d4a3a", "#6f8a86", "#2f6b52", "#d9a21b")
    p <- ggplot(df, aes(x = Metric, y = Value, text = paste0(Metric, ": ", round(Value, 2), " ", Unit))) +
      geom_col(fill = fills, width = 0.62) +
      coord_flip() + labs(x = NULL, y = "kg") + theme_forest()
    ggplotly(p, tooltip = "text") %>% config(displaylogo = FALSE)
  })
  
  output$tc_donut <- renderPlotly({
    df <- tc_values$results_df
    validate(need(!is.null(df), "No estimate yet."))
    plot_ly(labels = c("Above ground", "Below ground"), values = df$Value[3:4], type = "pie", hole = 0.62,
            marker = list(colors = c("#1d4a3a", "#d9a21b"), line = list(color = "#ffffff", width = 3)),
            textinfo = "percent", textfont = list(color = "#ffffff", size = 14, family = "Figtree, sans-serif"),
            hovertemplate = "%{label}: %{value:.1f} kg<extra></extra>") %>%
      layout(showlegend = TRUE, legend = list(orientation = "h", x = 0.5, xanchor = "center"),
             paper_bgcolor = "rgba(0,0,0,0)", font = list(family = "Figtree, sans-serif")) %>%
      config(displaylogo = FALSE)
  })
  
  output$tc_downloadResults <- downloadHandler(
    filename = function() paste0("Carbon_Stock_Results_", Sys.Date(), ".csv"),
    content = function(file) {
      req(tc_values$results_df)
      write.csv(tc_values$results_df, file, row.names = FALSE)
    }
  )
  
} # end server

# ============================================================
shinyApp(ui = ui, server = server)