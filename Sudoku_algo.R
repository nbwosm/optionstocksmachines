# Sudoku algorithm

## Create a function to check if a given 9x9 matrix solves a sudoku puzzle 


## Create function
sudoku_test <- function(mat_1){
  ## Current function assumes user inputs a standard 9x9 sudoku.
  
  ## Create correct sequence
  seq_1 <- seq(1:nrow(mat_1))
  
  ## Test rows
  count_r <- 0
  for(i in 1:nrow(mat_1)){
    row_sort <- sort(mat_1[i,])
    if(sum(row_sort == seq_1) < 9){
      print(paste("Row", i , "fails", sep=" "))
      break
    }else{
      count_r <- count_r+1
    }
  }
  
  stopifnot(count_r == nrow(mat_1))
  
  
  ## Test columns
  count_c <- 0
  for(i in 1:ncol(mat_1)){
    col_sort <- sort(mat_1[,i])
    if(sum(col_sort == seq_1) < 9){
      print(paste("Col", i, "fails", sep=" "))
      break
    }else{
      count_c <- count_c+1
    }
  }
  
  stopifnot(count_c == ncol(mat_1))
  
  ## Test 3 x 3 matrices
  start <- seq(1,9,sqrt(9))
  end <- start + sqrt(9) - 1
  
  row_start <- start
  row_end <- end
  col_start <- start
  col_end <- end
  
  count_i <- 0
  for(i in 1:3){
    count_j <- 0
    for(j in 1:3){
      out <- mat_1[row_start[i]:row_end[i], col_start[j]:col_end[j]]
      sort_out <- sort(out)
      if(sum(seq_1 == sort_out) < 9){
        print(paste("Matrix row", i, "col", j, "fails", sep = " "))
        break
      }else {
        count_j <- count_j + 1
      }
    }
    if(count_j < 3){
      print(paste("Matrix row", i, "col", j, "fails", sep = " "))
      break
    } else {
      count_i <- count_i + 1
    }
  }
  if(count_i < 3){
    print('Fail')
  } else {
    print('Success!')
  } 
}

## Import solved sudoku
sudo <- read.csv("sudoku.csv", header = FALSE, stringsAsFactors = FALSE)
sudo_mat <- as.matrix(sudo)
sudo_mat

# V1 V2 V3 V4 V5 V6 V7 V8 V9
# [1,]  7  2  6  4  9  3  8  1  5
# [2,]  3  1  5  7  2  8  9  4  6
# [3,]  4  8  9  6  5  1  2  3  7
# [4,]  8  5  2  1  4  7  6  9  3
# [5,]  6  7  3  9  8  5  1  2  4
# [6,]  9  4  1  3  6  2  7  5  8
# [7,]  1  9  4  8  3  6  5  7  2
# [8,]  5  6  7  2  1  4  3  8  9
# [9,]  2  3  8  5  7  9  4  6  1

## Generate unsolved sudoku
set.seed(42)
mat_a <- matrix(sample(9,81,replace = TRUE), byrow=TRUE, nrow=9)
mat_a

# [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    1    5    1    9    4    2    1    8    7
# [2,]    4    9    5    4    2    3    9    9    4
# [3,]    5    5    4    2    8    3    1    8    6
# [4,]    8    4    4    6    2    5    4    2    8
# [5,]    2    3    8    7    1    5    2    6    6
# [6,]    2    4    3    6    5    2    7    6    8
# [7,]    5    1    1    7    4    9    4    9    3
# [8,]    5    8    5    6    2    2    8    1    2
# [9,]    5    8    7    8    5    4    9    4    7

## Alter solved sudoku 
# First error does not appear until after row 4
set.seed(42)
row_samp <- sample(4:9, 3, replace = TRUE)
col_samp <- sample(9, 3, replace = TRUE)

mat_b <- sudo_mat

for(i in 1:3){
  for(j in 1:3){
    mat_b[row_samp[i], col_samp[j]] <- sample(9,1)
  }
}
mat_b

# V1 V2 V3 V4 V5 V6 V7 V8 V9
# [1,]  7  2  6  4  9  3  8  1  5
# [2,]  3  1  5  7  2  8  9  4  6
# [3,]  4  8  9  6  5  1  2  3  7
# [4,]  8  3  2  2  4  7  6  9  4
# [5,]  6  7  3  9  8  5  1  2  4
# [6,]  9  4  1  3  6  2  7  5  8
# [7,]  1  9  4  8  3  6  5  7  2
# [8,]  5  5  7  9  1  4  3  8  4
# [9,]  2  3  8  5  7  9  4  6  1

## Test function
sudoku_test(sudo_mat)
# Output
# [1] "Success!"

sudoku_test(mat_a)
# Output
# [1] "Row 1 fails"
# Error in sudoku_test(mat_a) : count_r == nrow(mat_1) is not TRUE 

sudoku_test(mat_b)
# Output
# [1] "Row 4 fails"
# Error in sudoku_test(mat_b) : count_r == nrow(mat_1) is not TRUE 