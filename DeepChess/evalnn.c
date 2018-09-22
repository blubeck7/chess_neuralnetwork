//  evalnn.c

#include "evalnn.h"

#include <math.h>
#include <stdio.h>
#include <Accelerate/Accelerate.h>

void load_nn_params(struct neural_net * nn)
{
        char *fn;
        FILE *fp;
        double tmp;
    
        // hidden layer weights
        fn = "/Users/brianlubeck/Documents/DataScience/springboard/capstone project 2/model/hw_weights";
        fp = fopen(fn, "rb");
        for (int i = 0; i < 768; i++) {
                for (int j = 0; j < 32000; j++) {
                        fread(&tmp, sizeof(double), 1, fp);
                        nn->hw[j * 768 + i] = (float) tmp;
                }
        }
        fclose(fp);
    
        // hidden layer biases
        fn = "/Users/brianlubeck/Documents/DataScience/springboard/capstone project 2/model/hb_weights";
        fp = fopen(fn, "rb");
        for (int i = 0; i < 32000; i++) {
                fread(&tmp, sizeof(double), 1, fp);
                nn->hb[i] = (float) tmp;
        }
        fclose(fp);
    
        // output layer weights
        fn = "/Users/brianlubeck/Documents/DataScience/springboard/capstone project 2/model/ow_weights";
        fp = fopen(fn, "rb");
        for (int i = 0; i < 32000; i++) {
                for (int j = 0; j < 3; j++) {
                        fread(&tmp, sizeof(double), 1, fp);
                        nn->ow[j * 32000 + i] = (float) tmp;
                }
        }
        fclose(fp);
    
        // output layer biases
        fn = "/Users/brianlubeck/Documents/DataScience/springboard/capstone project 2/model/ob_weights";
        fp = fopen(fn, "rb");
        for (int i = 0; i < 3; i++) {
                fread(&tmp, sizeof(double), 1, fp);
                nn->ob[i] = (float) tmp;
        }
        fclose(fp);
}

/*
 func:  evalnn
 desc:  calculates the score of a board position.
 args:  board = pointer to a 768 length array of either 1 or 0
        score = expected value of white win, draw or loss
        nn = neural net layer parameters
 ret:   0 if successful
 */
int evalnn(float * board, float * score, struct neural_net * nn)
{
        int success = -1;
        
        // hidden layer
        BNNSVectorDescriptor hid_in = {
                .data_type = BNNSDataTypeFloat32,
                .size = 768
        };
        
        BNNSVectorDescriptor hid_out = {
                .data_type = BNNSDataTypeFloat32,
                .size = 32000
        };
        
        BNNSFullyConnectedLayerParameters hid_params = {
                .in_size = 768,
                .out_size = 32000
        };
        
        hid_params.weights.data = nn->hw;
        hid_params.weights.data_type = BNNSDataTypeFloat32;

        hid_params.bias.data = nn->hb;
        hid_params.bias.data_type = BNNSDataTypeFloat32;
        
        hid_params.activation.function = BNNSActivationFunctionRectifiedLinear;
        
        BNNSFilter hid_filter;
        hid_filter = BNNSFilterCreateFullyConnectedLayer(
                &hid_in,
                &hid_out,
                &hid_params,
                NULL
        );
        
        float hid[32000];
        success = BNNSFilterApply(hid_filter, board, hid);

        if (success == -1)
                return success;
        
        // output layer
        BNNSVectorDescriptor out_in = {
                .data_type = BNNSDataTypeFloat32,
                .size = 32000
        };
        
        BNNSVectorDescriptor out_out = {
                .data_type = BNNSDataTypeFloat32,
                .size = 3
        };
        
        BNNSFullyConnectedLayerParameters out_params = {
                .in_size = 32000,
                .out_size = 3
        };
        
        out_params.weights.data = nn->ow;
        out_params.weights.data_type = BNNSDataTypeFloat32;
        
        out_params.bias.data = nn->ob;
        out_params.bias.data_type = BNNSDataTypeFloat32;
        
        out_params.activation.function = BNNSActivationFunctionIdentity;
        
        BNNSFilter out_filter;
        out_filter = BNNSFilterCreateFullyConnectedLayer(
                &out_in,
                &out_out,
                &out_params,
                NULL
        );
        
        float probs[3];
        success = BNNSFilterApply(out_filter, hid, probs);
        
        //softmax
        float max_logit;
        float shifted_probs[3];
        float tot = 0;
        max_logit = probs[0];
        for (int i = 1; i < 3; i ++)
                if (probs[i] > max_logit)
                        max_logit = probs[i];
        
        for (int i = 0; i < 3; i++) {
                shifted_probs[i] = exp(probs[i] - max_logit);
                tot += shifted_probs[i];
        }
        
        for (int i = 0; i < 3; i++)
                probs[i] = shifted_probs[i] / tot;

        *score = probs[0] - probs[2];
        
    return success;
}
