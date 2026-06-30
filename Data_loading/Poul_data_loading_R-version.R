library(readr)
library(dplyr)
library(zoo)
library(openxlsx)
library(stringr)
library(dotenv)

# Load environment variables from .env file
load_dot_env()

# Define paths
path      <- Sys.getenv("Pouls_data_path")
save_path <- Sys.getenv("Pouls_save_path")

# --- Load and process each file ---
all_files <- sort(list.files(path, pattern = "^R", full.names = TRUE))

# Interpolation algorithm: Attempt to replicate Python's Pandas' interpolate method
interpolate_zeros <- function(s) {
  
  s <- as.numeric(s)
  n <- length(s)
  
  artifact <- rep(FALSE, n)
  
  # Python: s[i] == 0
  artifact[s == 0] <- TRUE
  
  # Python: 0 < i < n-1 and s[i-1]==0 and s[i+1]==0
  if (n >= 3) {
    center <- 2:(n-1)
    artifact[center] <- artifact[center] |
      (s[center - 1] == 0 & s[center + 1] == 0)
  }
  
  s[artifact] <- NA
  
  # pandas interpolate(method='linear') default behavior:
  # - leading NaNs stay NaN
  # - trailing NaNs are filled with the last valid value
  s <- zoo::na.approx(s, rule = c(1, 2), na.rm = FALSE)
  
  round(s, 2)
}

# Define lists to contain the cleaned data and removed rows respectively
all_dfs <- list()
dropped_rows_list <- list()

for (file in all_files){
  filename <- str_extract(file, "[^/]+$")
  
  # 
  cat("Preprocessing file:", filename, "\n")
  df <- read.delim(file, skip = 1, sep = "\t", stringsAsFactors = FALSE)
  
  # Select only relevant protocol type and remove redundant rows
  df <- df %>% filter(Protocol.Type == "PLR-Positive")
  df <- df %>% select(DateTime, PatientID, Pupil.Measured, 25:803)

  # Define time-series columns
  ts_cols <- names(df)[c(4:length(df))]
  
  # Add a stable row ID for this file, used only for tracking drops
  df$row_id <- paste(filename, seq_len(nrow(df)), sep = "_")
  df$source_file <- filename
  
  # Removing recordings with patients having a non-standard Danish ID
  before <- df
  df <- df %>% filter(str_detect(str_extract(PatientID, "^[^.]+"), "^\\d+$"))
  df <- df %>% filter(nchar(as.character(PatientID)) > 6)
  dropped <- before %>% filter(!row_id %in% df$row_id)
  if (nrow(dropped) > 0) {
    dropped$reason <- "non-numeric or short PatientID"
    dropped$source_file <- filename
    dropped_rows_list[[length(dropped_rows_list) + 1]] <- dropped
  }
  cat("Discarded", nrow(before) - nrow(df), "rows with non-numeric PatientID in file:", filename, "\n")
  
  # Make sure all dates are same format. OBS. Some measurements have seconds, those are kept.
  # We convert "-" to "/" and change all format of years to XX, e.g. 01/01/2026 -> 01/01/26.
  df <- df %>%
    mutate(
      DateTime = str_replace_all(DateTime, "-", "/"),
      DatePart = str_split_fixed(DateTime, " ", 2)[,1],
      TimePart = str_split_fixed(DateTime, " ", 2)[,2],
      DateTime = paste0(
        substr(DatePart, 1, 6),
        substr(DatePart, nchar(DatePart) - 1, nchar(DatePart)),
        " ",
        TimePart
      )
    ) %>%
    select(-DatePart, -TimePart)
  
  # Zero-padding to ensure all patient ids have length 10:
  df$PatientID <- str_pad(as.character(df$PatientID), width = 10, side = "left", pad = "0")
  
  # Discarding rows with at least first 8 initial values being 0
  before <- df
  df <- df[apply(df[, ts_cols[1:8]], 1, function(x) max(x, na.rm = TRUE) > 0), ]
  dropped <- before %>% filter(!row_id %in% df$row_id)
  if (nrow(dropped) > 0) {
    dropped$reason <- "first 8 values all zero"
    dropped$source_file <- filename
    dropped_rows_list[[length(dropped_rows_list) + 1]] <- dropped
  }
  cat("Discarded", nrow(before) - nrow(df), "rows due to all first 8 values being zero in file:", filename, "\n")

  # Interpolation:
  df[ts_cols] <- t(apply(df[ts_cols], 1, interpolate_zeros))
  
  # Discarding duplicate rows matching on Datetime, PatientID and Pupil.Measured
  before <- df
  df <- df %>% distinct(DateTime, PatientID, Pupil.Measured, .keep_all = TRUE)
  dropped <- before %>% filter(!row_id %in% df$row_id)
  if (nrow(dropped) > 0) {
    dropped$reason <- "duplicate within file"
    dropped$source_file <- filename
    dropped_rows_list[[length(dropped_rows_list) + 1]] <- dropped
  }
  cat("Discarded", nrow(before) - nrow(df), "duplicate rows in file: ", filename, "\n")
  
  # Append to list with all dataframes
  all_dfs[[length(all_dfs) + 1]] <- df
}

# Combine all dataframes
combined_df <- bind_rows(all_dfs)

# Discarding duplicate rows matching on Datetime, PatientID and Pupil.Measured across all files
before <- combined_df
combined_df <- combined_df %>% distinct(DateTime, PatientID, Pupil.Measured, .keep_all = TRUE)
dropped_cross <- before %>% filter(!row_id %in% combined_df$row_id)
if (nrow(dropped_cross) > 0) {
  dropped_cross$reason <- "duplicate across files"
  dropped_rows_list[[length(dropped_rows_list) + 1]] <- dropped_cross
}
cat("Discarded", nrow(before) - nrow(combined_df), "duplicate rows across all files.\n")

# Combine all dropped rows into one dataframe
dropped_df <- bind_rows(dropped_rows_list)

# Removing "helper row id"
combined_df <- combined_df %>% select(-row_id)

# Saving dropped data dataframe to an Excel sheet
write.xlsx(
  dropped_df,
  file = file.path(save_path, "Poul_dropped_data_from_R.xlsx")
)

# Saving combined dataframe to an Excel sheet
write.xlsx(
  combined_df,
  file = file.path(save_path, "Poul_data_cleaned_from_R.xlsx")
)

