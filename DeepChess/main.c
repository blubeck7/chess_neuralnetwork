//main.c

#include "evalnn.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char * argv[])
{
        static struct neural_net nn;
        load_nn_params(&nn);
        
        float board[768] = {0};
        float score = 0;
        board[88] = 1;
        board[100] = 1;
        board[200] = 1;
        board[300] = 1;
        board[400] = 1;
        board[500] = 1;
        board[542] = 1;
        
        int success = evalnn(board, &score, &nn);
        
        printf("Success %d\n", success);
        printf("Score %f\n", score);
        
        return 0;
}
