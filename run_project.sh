#!/usr/bin/env bash

# Enhanced setup_and_run.sh — Interactive CLIF Sedation Project execution script (Mac/Linux)

# ── ANSI colours ─────────────────────────────────────────────
YELLOW="\033[33m"
CYAN="\033[36m"
GREEN="\033[32m"
RED="\033[31m"
BLUE="\033[34m"
BOLD="\033[1m"
RESET="\033[0m"

# ── Setup logging ──────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="${SCRIPT_DIR}/logs/execution_log_${TIMESTAMP}.log"
mkdir -p "${SCRIPT_DIR}/logs"

# Function to log and display
log_echo() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Initialize log file with header
echo "CLIF Epidemiology of Sedation Project - Execution Log" > "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

show_banner() {
    clear
    log_echo "${CYAN}${BOLD}"
    log_echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    log_echo "║                                                                              ║"
    log_echo "║                              ██████ ██      ██ ███████                       ║"
    log_echo "║                             ██      ██      ██ ██                            ║"
    log_echo "║                             ██      ██      ██ █████                         ║"
    log_echo "║                             ██      ██      ██ ██                            ║"
    log_echo "║                              ██████ ███████ ██ ██                            ║"
    log_echo "║                                                                              ║"
    log_echo "║                        EPIDEMIOLOGY OF SEDATION PROJECT                      ║"
    log_echo "║                                                                              ║"
    log_echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    log_echo "${RESET}"
    log_echo ""
    log_echo "${GREEN}Welcome to the CLIF Epidemiology of Sedation Project!${RESET}"
    log_echo ""
}

# ── Progress separator ─────────────────────────────────────────────────────────
separator() {
    log_echo "${YELLOW}==================================================${RESET}"
}

# ── Progress bar function ──────────────────────────────────────────────────────
show_progress() {
    local step=$1
    local total=$2
    local description=$3
    
    separator
    log_echo "${CYAN}${BOLD}Step ${step}/${total}: ${description}${RESET}"
    log_echo "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${description}${RESET}"
    separator
}

# ── Error handler ──────────────────────────────────────────────────────────────
# Global array to track failed steps
FAILED_STEPS=()

handle_error() {
    local exit_code=$?
    local step_name=$1
    
    log_echo ""
    log_echo "${RED}${BOLD}❌ ERROR OCCURRED!${RESET}"
    log_echo "${RED}Step failed: ${step_name}${RESET}"
    log_echo "${RED}Exit code: ${exit_code}${RESET}"
    log_echo "${RED}Check the log file for full details: ${LOG_FILE}${RESET}"
    log_echo "${YELLOW}Continuing with next step...${RESET}"
    log_echo ""
    
    # Add to failed steps array
    FAILED_STEPS+=("$step_name")
    
    return 0  # Continue execution
}

# Export environment variables for unbuffered output
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

# ── Main execution flow ────────────────────────────────────────────────────────
show_banner

# Step 1: Create virtual environment
show_progress 1 5 "Create Virtual Environment"
if [ ! -d ".venv" ]; then
    log_echo "Creating virtual environment (.venv)..."
    python3 -m venv .venv 2>&1 | tee -a "$LOG_FILE" || handle_error "Create Virtual Environment"
else
    log_echo "Virtual environment already exists."
fi
log_echo "${GREEN}✅ Completed: Create Virtual Environment${RESET}"

# Step 2: Activate virtual environment
show_progress 2 5 "Activate Virtual Environment"
log_echo "Activating virtual environment..."
source .venv/bin/activate || handle_error "Activate Virtual Environment"
log_echo "${GREEN}✅ Completed: Activate Virtual Environment${RESET}"

# Step 3: Install dependencies
show_progress 3 5 "Install Dependencies"
log_echo "Upgrading pip..."
python -m pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE" || handle_error "Upgrade pip"

log_echo "Installing dependencies..."
pip install -r requirements.txt 2>&1 | tee -a "$LOG_FILE" || handle_error "Install requirements"
pip install jupyter ipykernel 2>&1 | tee -a "$LOG_FILE" || handle_error "Install jupyter"
log_echo "${GREEN}✅ Completed: Install Dependencies${RESET}"

# Step 4: Register Jupyter kernel
show_progress 4 5 "Register Jupyter Kernel"
log_echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=.venv --display-name="Python (sedation)" 2>&1 | tee -a "$LOG_FILE" || handle_error "Register Jupyter Kernel"
log_echo "${GREEN}✅ Completed: Register Jupyter Kernel${RESET}"

# Step 5: Validate configuration and data
show_progress 5 5 "Validate Configuration & Data"
log_echo "Checking current directory structure..."

# Check if data path exists
log_echo "Checking data configuration..."
CONFIG_FILE="config/config.json"
if [ -f "$CONFIG_FILE" ]; then
    DATA_PATH=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['tables_path'])" 2>/dev/null)
    if [ -n "$DATA_PATH" ]; then
        if [ -d "$DATA_PATH" ]; then
            log_echo "${GREEN}✅ Data path found: $DATA_PATH${RESET}"
        else
            log_echo "${YELLOW}⚠️  Data path not found: $DATA_PATH${RESET}"
            log_echo "${YELLOW}Please ensure the CLIF data is available at this location${RESET}"
            log_echo "${YELLOW}Or update config/config.json with the correct path${RESET}"
            log_echo ""
            read -p "Continue anyway? (y/n): " continue_choice
            if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
                log_echo "${BLUE}Exiting. Please set up the data path and try again.${RESET}"
                exit 0
            fi
        fi
    fi
else
    log_echo "${YELLOW}⚠️  Config file not found: $CONFIG_FILE${RESET}"
    log_echo "${YELLOW}Please copy config/config_template.json to config/config.json and update it${RESET}"
fi

log_echo "${GREEN}✅ Completed: Validate Configuration & Data${RESET}"

# Final summary
separator
log_echo "${CYAN}${BOLD}📋 SETUP COMPLETE${RESET}"
separator

# Display success/failure summary
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    log_echo "${GREEN}${BOLD}🎉 SUCCESS! Environment setup completed successfully!${RESET}"
else
    log_echo "${YELLOW}${BOLD}⚠️  PARTIAL SUCCESS: Some steps failed${RESET}"
    log_echo ""
    log_echo "${RED}${BOLD}Failed steps:${RESET}"
    for step in "${FAILED_STEPS[@]}"; do
        log_echo "${RED}  ❌ $step${RESET}"
    done
    log_echo ""
    log_echo "${YELLOW}Please check the log file for error details: ${LOG_FILE}${RESET}"
fi

log_echo ""
separator
log_echo "${CYAN}${BOLD}📝 NEXT STEPS${RESET}"
separator
log_echo ""
log_echo "${BOLD}To run the analysis:${RESET}"
log_echo ""
log_echo "  ${CYAN}1.${RESET} Open your IDE (VS Code, PyCharm, etc.)"
log_echo "  ${CYAN}2.${RESET} Select the ${BOLD}.venv${RESET} Python interpreter"
log_echo "  ${CYAN}3.${RESET} Open: ${BOLD}code/sedation_sbt.ipynb${RESET}"
log_echo "  ${CYAN}4.${RESET} Run the notebook interactively"
log_echo ""
log_echo "${BLUE}📝 Setup log saved to: ${LOG_FILE}${RESET}"
separator
log_echo ""
log_echo "${GREEN}Thank you for using the CLIF Epidemiology of Sedation Project!${RESET}"
log_echo ""
read -rp "Press [Enter] to exit..."