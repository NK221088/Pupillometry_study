library(tidyverse)
library(writexl)
library(zoo)

path      <- "L:/Auditdata/CONNECT-ME/Pupillometry to detect covert awareness/Pupillometridata/CSV_format"
save_path <- "L:/Auditdata/CONNECT-ME/Pupillometry to detect covert awareness/Pupillometridata"

interpolate_zeros_matrix <- function(mat) {
  mat[mat == 0] <- NA
  is_na <- is.na(mat)
  left  <- cbind(FALSE, is_na[, -ncol(mat)])
  right <- cbind(is_na[, -1], FALSE)
  mat[left & right] <- NA
  t(apply(mat, 1, function(row) {
    round(zoo::na.approx(row, na.rm = FALSE, rule = 2), 2)
  }))
}

all_files <- list.files(path, pattern = "^R", full.names = TRUE) |> sort()

dfs <- list()

for (file in all_files) {
  cat("Processing:", file, "\n")
  
  df <- read_tsv(file, skip = 1, show_col_types = FALSE)
  df <- df |> filter(`Protocol-Type` == "PLR-Positive")
  
  ts_cols   <- names(df)[25:803]
  keep_cols <- c("DateTime", "PatientID", "Pupil-Measured", ts_cols)
  df        <- df |> select(all_of(keep_cols))
  
  len_before <- nrow(df)
  df <- df |>
    filter(str_split_i(as.character(PatientID), fixed("."), 1) |> str_detect("^\\d+$")) |>
    mutate(PatientID = as.integer(as.numeric(PatientID))) |>
    filter(nchar(as.character(PatientID)) > 6) |>
    mutate(PatientID = str_pad(as.character(PatientID), 10, pad = "0"))
  cat(sprintf("Discarded %d rows with non-numeric PatientID in file: %s\n",
              len_before - nrow(df), basename(file)))
  
  ts_matrix   <- as.matrix(df[ts_cols])
  ts_matrix   <- suppressWarnings(apply(ts_matrix, 2, as.numeric))
  ts_matrix   <- interpolate_zeros_matrix(ts_matrix)
  df[ts_cols] <- as.data.frame(ts_matrix)
  
  len_before <- nrow(df)
  df <- df |> filter(apply(df[ts_cols[1:8]], 1, max, na.rm = TRUE) > 0)
  cat(sprintf("Discarded %d rows due to all first 8 values being zero in file: %s\n",
              len_before - nrow(df), basename(file)))
  
  len_before <- nrow(df)
  df <- df |> distinct(DateTime, PatientID, `Pupil-Measured`, .keep_all = TRUE)
  cat(sprintf("Discarded %d duplicate rows in file: %s\n",
              len_before - nrow(df), basename(file)))
  
  dfs[[length(dfs) + 1]] <- df
}

df_all <- bind_rows(dfs)

len_before <- nrow(df_all)
df_all     <- df_all |> distinct(DateTime, PatientID, `Pupil-Measured`, .keep_all = TRUE)
cat(sprintf("Discarded %d duplicate rows across all files.\n", len_before - nrow(df_all)))

df_all <- df_all |>
  mutate(
    dt_parts  = str_replace_all(DateTime, "-", "/"),
    date_part = str_split_i(dt_parts, " ", 1),
    time_part = str_split_i(dt_parts, " ", -1),
    year2     = str_sub(str_split_i(date_part, "/", -1), -2),
    DateTime  = paste0(str_sub(date_part, 1, 6), year2, " ", time_part)
  ) |>
  select(-dt_parts, -date_part, -time_part, -year2)

write_xlsx(df_all, file.path(save_path, "Poul_data_cleaned.xlsx"))