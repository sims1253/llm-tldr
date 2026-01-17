# Test file for R language support in TLDR

# Simple function with control flow
calculate_sum <- function(n) {
  total <- 0
  i <- 1

  if (n > 0) {
    while (i <= n) {
      total <- total + i
      i <- i + 1

      if (total > 100) {
        break
      }
    }
  } else {
    total <- 0
  }

  return(total)
}

# Function with for loop
process_list <- function(items) {
  result <- c()

  for (item in items) {
    if (item > 10) {
      result <- c(result, item)
    } else {
      next # R's continue equivalent
    }
  }

  return(result)
}

# Function with repeat loop
count_down <- function(start) {
  x <- start

  repeat {
    x <- x - 1
    if (x <= 0) break
  }

  return(x)
}

# Function with switch statement
get_status <- function(code) {
  switch(
    code,
    "1" = {
      "OK"
    },
    "2" = {
      "Warning"
    },
    "3" = {
      "Error"
    },
    {
      "Unknown"
    }
  )
}
