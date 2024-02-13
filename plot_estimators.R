
library(ggplot2)
library(hrbrthemes)
library(dplyr)
library(forcats)
library(RColorBrewer)
library(rcartocolor)
# Function to create a plot




# Function to create a plot
createPlot <- function(data, plot_type, species_name) {
  if (!"TRAIT" %in% colnames(data)) {
    cat("The 'TRAIT' variable is not found in the dataset for species:", species_name, "\n")
    return(NULL)
  }
  
  # Convert markers column to character
  data$markers <- as.character(data$markers)
  
  p <- ggplot(data, aes(x = markers, y = .data[[plot_type]], fill = as.character(estimators))) +
    geom_boxplot() +  
    theme(axis.text.x = element_text(angle = 90, hjust = 1), panel.grid.major = element_blank(),panel.grid.minor = element_blank()) +
    facet_grid(as.factor(TRAIT) ~ ., scales = "free_y") + xlab("No. markers")+
    ggtitle(species_name)  +
    theme_classic()+
    scale_fill_brewer(palette = "Set1")  # Use RColorBrewer for a discrete color palette
  
  # Print and save the plot as before
  
  # Print the plot
  print(p+labs(fill = "estimators"))
  
  # Save the plot to a PDF with the species name in the filename
  pdf_file <- sprintf("/Users/mariaestefania/Documents/ETH/python/euler/GBR_results/%s_Plot_%s.pdf", plot_type, species_name)
  pdf(file = pdf_file, width = 10, height = 20)
  print(p+labs(fill = "estimators"))
  dev.off()
}


# List all files in the directory
FILES <- list.files()
FILES

# Read species names from a text file (assuming the species names are in one column)
SPECIES <- read.table('/Users/mariaestefania/Documents/ETH/python/euler/scripts/species.txt', header = FALSE, stringsAsFactors = FALSE)
list_species <- SPECIES$V1  # Use the column name to extract the species names




for (i in list_species) {
  i <- trimws(i)
  
  cat("Species:", i, "\n")
  
  file1 <- FILES[grep(i, FILES, fixed = TRUE)]
  file1 <- file1[grep('csv', file1)]
  cat("Matching files for", i, ":", file1, "\n")
  
  data <- NULL
  
  for (x in file1) {
    # Check if the file exists before attempting to read
    if (file.exists(x)) {
      temp_data <- tryCatch({
        read.csv(x)
      }, error = function(e) {
        cat("Error reading file:", x, " - Error:", conditionMessage(e), "\n")
        return(NULL)
      })
      
      if (!is.null(temp_data)) {
        temp_data$Species <- i
        data <- bind_rows(data, temp_data)
      }
    } else {
      cat("File does not exist:", x, "\n")
    }
  }
  
  # Create ggplot visualizations for correlation (cor) and root mean square error (rmse)
  createPlot(data, "cor", i)
  createPlot(data, "rmse", i)
  
}
