//  evalnn.h

#ifndef EVALNN_H
#define EVALNN_H

struct neural_net {
        float hw[768 * 32000];
        float hb[32000];
        float ow[32000 * 3];
        float ob[3];
};

void load_nn_params(struct neural_net * nn);
int evalnn(float * board, float * score, struct neural_net * nn);
#endif /* EVALNN_H */
