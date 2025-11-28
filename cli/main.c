#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "maf.h"

#define MAX_LINE_LEN 65536
#define MAX_TOKENS 4096

/* ========================================================================== */
/* CSV Utilities                                                              */
/* ========================================================================== */

typedef struct {
    float* data;
    int rows;
    int cols;
} Dataset;

void free_dataset(Dataset* ds) {
    if (ds) {
        free(ds->data);
        free(ds);
    }
}

Dataset* load_csv(const char* filename, bool skip_header) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    char line[MAX_LINE_LEN];
    int rows = 0;
    int cols = 0;
    
    /* Pass 1: count rows and cols */
    while (fgets(line, sizeof(line), f)) {
        if (skip_header && rows == 0) {
            rows++; /* Count header as row for logic, but ignore */
            continue;
        }
        if (cols == 0) {
            char* tmp = strdup(line);
            char* token = strtok(tmp, ",");
            while (token) {
                cols++;
                token = strtok(NULL, ",");
            }
            free(tmp);
        }
        rows++;
    }
    rewind(f);

    int actual_rows = skip_header ? rows - 1 : rows;
    if (actual_rows <= 0 || cols <= 0) {
        fprintf(stderr, "Error: Empty or invalid CSV %s\n", filename);
        fclose(f);
        return NULL;
    }

    Dataset* ds = malloc(sizeof(Dataset));
    ds->rows = actual_rows;
    ds->cols = cols;
    ds->data = malloc((size_t)actual_rows * cols * sizeof(float));

    int r = 0;
    int line_idx = 0;
    while (fgets(line, sizeof(line), f)) {
        if (skip_header && line_idx == 0) {
            line_idx++;
            continue;
        }
        
        /* Remove newline */
        line[strcspn(line, "\r\n")] = 0;
        
        char* token = strtok(line, ",");
        int c = 0;
        while (token && c < cols) {
            ds->data[r * cols + c] = strtof(token, NULL);
            token = strtok(NULL, ",");
            c++;
        }
        if (c != cols) {
            fprintf(stderr, "Warning: Row %d has %d columns, expected %d\n", r, c, cols);
        }
        r++;
        line_idx++;
    }

    fclose(f);
    return ds;
}

/* ========================================================================== */
/* Model Serialization                                                        */
/* ========================================================================== */

#define MAGIC "MAF1"

void save_model(const char* filename, maf_model_t* model) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open output file %s\n", filename);
        exit(1);
    }

    /* Header */
    fwrite(MAGIC, 1, 4, f);
    fwrite(&model->n_flows, sizeof(uint16_t), 1, f);
    fwrite(&model->param_dim, sizeof(uint16_t), 1, f);
    fwrite(&model->feature_dim, sizeof(uint16_t), 1, f);
    
    /* Assume constant hidden size for now, take from first layer */
    uint16_t hidden = model->layers[0].hidden_units;
    fwrite(&hidden, sizeof(uint16_t), 1, f);

    /* Data */
    int n = model->n_flows;
    int D = model->param_dim;
    int C = model->feature_dim;
    int H = hidden;

    /* Iterate layers and write blocks */
    /* Structure needs to match maf_weights_t expectation of flat arrays */
    /* So we write all M1s, then all M2s, etc. */

    /* M1 */
    for(int k=0; k<n; k++) fwrite(model->layers[k].M1, sizeof(float), H*D, f);
    /* M2 */
    for(int k=0; k<n; k++) fwrite(model->layers[k].M2, sizeof(float), D*H, f);
    /* Perm */
    for(int k=0; k<n; k++) fwrite(model->layers[k].perm, sizeof(uint16_t), D, f);
    /* InvPerm */
    for(int k=0; k<n; k++) fwrite(model->layers[k].inv_perm, sizeof(uint16_t), D, f);
    /* W1y */
    for(int k=0; k<n; k++) fwrite(model->layers[k].W1y, sizeof(float), H*D, f);
    /* W1c */
    for(int k=0; k<n; k++) fwrite(model->layers[k].W1c, sizeof(float), H*C, f);
    /* b1 */
    for(int k=0; k<n; k++) fwrite(model->layers[k].b1, sizeof(float), H, f);
    /* W2 */
    for(int k=0; k<n; k++) fwrite(model->layers[k].W2, sizeof(float), 2*D*H, f);
    /* W2c */
    for(int k=0; k<n; k++) fwrite(model->layers[k].W2c, sizeof(float), 2*D*C, f);
    /* b2 */
    for(int k=0; k<n; k++) fwrite(model->layers[k].b2, sizeof(float), 2*D, f);

    fclose(f);
    printf("Saved model to %s\n", filename);
}

/* Global buffer pointers to keep alive for maf_weights_t */
float *g_M1, *g_M2, *g_W1y, *g_W1c, *g_b1, *g_W2, *g_W2c, *g_b2;
uint16_t *g_perm, *g_inv_perm;

maf_model_t* load_model_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open model file %s\n", filename);
        exit(1);
    }

    char magic[5] = {0};
    if (fread(magic, 1, 4, f) != 4) goto error;
    if (strcmp(magic, MAGIC) != 0) {
        fprintf(stderr, "Error: Invalid file format\n");
        fclose(f);
        exit(1);
    }

    uint16_t n_flows, param_dim, feature_dim, hidden;
    if (fread(&n_flows, sizeof(uint16_t), 1, f) != 1) goto error;
    if (fread(&param_dim, sizeof(uint16_t), 1, f) != 1) goto error;
    if (fread(&feature_dim, sizeof(uint16_t), 1, f) != 1) goto error;
    if (fread(&hidden, sizeof(uint16_t), 1, f) != 1) goto error;

    int n = n_flows;
    int D = param_dim;
    int C = feature_dim;
    int H = hidden;

    /* Allocate flat buffers */
    g_M1 = malloc(n * H * D * sizeof(float));
    g_M2 = malloc(n * D * H * sizeof(float));
    g_perm = malloc(n * D * sizeof(uint16_t));
    g_inv_perm = malloc(n * D * sizeof(uint16_t));
    g_W1y = malloc(n * H * D * sizeof(float));
    g_W1c = malloc(n * H * C * sizeof(float));
    g_b1 = malloc(n * H * sizeof(float));
    g_W2 = malloc(n * 2 * D * H * sizeof(float));
    g_W2c = malloc(n * 2 * D * C * sizeof(float));
    g_b2 = malloc(n * 2 * D * sizeof(float));

    if (fread(g_M1, sizeof(float), n*H*D, f) != n*H*D) goto error;
    if (fread(g_M2, sizeof(float), n*D*H, f) != n*D*H) goto error;
    if (fread(g_perm, sizeof(uint16_t), n*D, f) != n*D) goto error;
    if (fread(g_inv_perm, sizeof(uint16_t), n*D, f) != n*D) goto error;
    if (fread(g_W1y, sizeof(float), n*H*D, f) != n*H*D) goto error;
    if (fread(g_W1c, sizeof(float), n*H*C, f) != n*H*C) goto error;
    if (fread(g_b1, sizeof(float), n*H, f) != n*H) goto error;
    if (fread(g_W2, sizeof(float), n*2*D*H, f) != n*2*D*H) goto error;
    if (fread(g_W2c, sizeof(float), n*2*D*C, f) != n*2*D*C) goto error;
    if (fread(g_b2, sizeof(float), n*2*D, f) != n*2*D) goto error;

    fclose(f);

    /* Setup weights struct */
    maf_weights_t w;
    w.n_flows = n;
    w.param_dim = D;
    w.feature_dim = C;
    w.hidden_units = H;
    w.M1_data = g_M1;
    w.M2_data = g_M2;
    w.perm_data = g_perm;
    w.inv_perm_data = g_inv_perm;
    w.W1y_data = g_W1y;
    w.W1c_data = g_W1c;
    w.b1_data = g_b1;
    w.W2_data = g_W2;
    w.W2c_data = g_W2c;
    w.b2_data = g_b2;

    return maf_load_model(&w);

error:
    fprintf(stderr, "Error: Unexpected end of file or read error\n");
    fclose(f);
    exit(1);
}

/* ========================================================================== */
/* Random Model Initialization                                                */
/* ========================================================================== */

float rand_f() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

maf_model_t* init_random_model(int n_flows, int D, int C, int H) {
    int n = n_flows;
    
    /* Use global buffers for simplicity, similar to load */
    g_M1 = malloc(n * H * D * sizeof(float));
    g_M2 = malloc(n * D * H * sizeof(float));
    g_perm = malloc(n * D * sizeof(uint16_t));
    g_inv_perm = malloc(n * D * sizeof(uint16_t));
    g_W1y = malloc(n * H * D * sizeof(float));
    g_W1c = malloc(n * H * C * sizeof(float));
    g_b1 = malloc(n * H * sizeof(float));
    g_W2 = malloc(n * 2 * D * H * sizeof(float));
    g_W2c = malloc(n * 2 * D * C * sizeof(float));
    g_b2 = malloc(n * 2 * D * sizeof(float));

    /* Masks: Random binary for now or MADE structure? 
       MAF usually uses random masking that satisfies MADE property.
       For simplicity here, we'll use full connectivity mask=1 (not autoregressive) 
       Wait, MAF MUST be autoregressive.
       Simplest autoregressive mask: m[k] <= k.
       Let's implement a basic sequential mask builder.
    */
    
        /* Initialize proper MADE masks */
    
        for (int k=0; k<n; k++) {
    
            /* 1. Create Permutation */
    
            for (int i=0; i<D; i++) g_perm[k*D+i] = i;
    
            /* Shuffle permutation */
    
            for (int i=0; i<D; i++) {
    
                int j = rand() % D;
    
                uint16_t tmp = g_perm[k*D+i];
    
                g_perm[k*D+i] = g_perm[k*D+j];
    
                g_perm[k*D+j] = tmp;
    
            }
    
            /* Compute inverse */
    
            for (int i=0; i<D; i++) g_inv_perm[k*D + g_perm[k*D+i]] = i;
    
    
    
            /* 2. Assign Degrees */
    
            /* Inputs: m[i] = i+1 (1..D) */
    
            /* Hidden: m_h[j] sample from 1..(D-1) */
    
            /* Outputs: m[d] = d+1 (1..D) */
    
            
    
            int* m_in = malloc(D * sizeof(int));
    
            int* m_h = malloc(H * sizeof(int));
    
            
    
            for(int i=0; i<D; i++) m_in[i] = i + 1; /* 1-based degrees */
    
            
    
            for(int j=0; j<H; j++) {
    
                 /* Random degree between 1 and D-1 */
    
                 /* If D=1, MADE is trivial (no connectivity for autoregression usually implies D>1)
    
                    If D=1, we can't have m_h < m_out. 
    
                    But standard MAF/MADE works for D>=1? 
    
                    If D=1, p(x) is just conditional on context. No autoregression over 1 dim.
    
                    Masks should allow everything if D=1?
    
                    Standard MADE logic: sample from 1..D-1. If D=1, range is empty.
    
                    Let's handle D=1: m_h=0? 
    
                    If D=1, input degree 1, output degree 1. 
    
                    Condition m_h < m_out => m_h < 1 => m_h=0.
    
                    Condition m_in <= m_h => 1 <= 0 False.
    
                    So for D=1, no path from input to output via hidden?
    
                    Correct. For 1D, flow is just x -> f(x). 
    
                    Wait, MAF transforms z to x.
    
                    u = (x - mu) * exp(-alpha). mu, alpha depend on previous x.
    
                    For x_0, mu, alpha depend on NOTHING (or context).
    
                    So connectivity from x to mu/alpha must be ZERO.
    
                    So for D=1, M1/M2 should effectively cut connection x->h->out.
    
                    So standard logic holds.
    
                 */
    
                 if (D > 1) m_h[j] = (rand() % (D - 1)) + 1;
    
                 else m_h[j] = 0; 
    
            }
    
    
    
            /* 3. Build Masks */
    
            /* M1: H x D. M1[j, i] = 1 if m_in[i] <= m_h[j] */
    
            for (int j=0; j<H; j++) {
    
                for (int i=0; i<D; i++) {
    
                    /* Note: g_M1 is row-major [j*D + i] */
    
                    if (m_in[i] <= m_h[j]) g_M1[k*H*D + j*D + i] = 1.0f;
    
                    else g_M1[k*H*D + j*D + i] = 0.0f;
    
                    
    
                                    /* Initialize Weights */
    
                    
    
                                    g_W1y[k*H*D + j*D + i] = rand_f() * 0.01f;
    
                    
    
                                }
    
                    
    
                                /* Initialize Biases and Context Weights */
    
                    
    
                                for (int c=0; c<C; c++) g_W1c[k*H*C + j*C + c] = rand_f() * 0.01f;
    
                g_b1[k*H + j] = 0.0f;
    
            }
    
    
    
            /* M2: 2D x H. M2[d, j] = 1 if m_h[j] < m_out[d] */
    
            /* Output degrees match input degrees 1..D */
    
            for (int d=0; d<D; d++) {
    
                int m_out = d + 1;
    
                /* M2 is for both mu and alpha, concatenated? 
    
                   maf_layer_t has W2 [2*D * H]. 
    
                   Output is size 2*D. First D are mu, second D are alpha.
    
                   Both respect the same degree constraints.
    
                */
    
                
    
                /* Iterate hidden units */
    
                for (int j=0; j<H; j++) {
    
                    float val = (m_h[j] < m_out) ? 1.0f : 0.0f;
    
                    
    
                    /* Set for M2 (shared mask for mu and alpha, size D*H) */
    
                    g_M2[k*D*H + d*H + j] = val;
    
                    
    
                    /* Weights */
    
                    g_W2[k*2*D*H + d*H + j] = rand_f() * 0.01f;
    
                    g_W2[k*2*D*H + (D+d)*H + j] = rand_f() * 0.01f;
    
                }
    
                
    
                /* Biases and Context */
    
                for(int c=0; c<C; c++) {
    
                    g_W2c[k*2*D*C + d*C + c] = rand_f() * 0.01f;
    
                    g_W2c[k*2*D*C + (D+d)*C + c] = rand_f() * 0.01f;
    
                }
    
                g_b2[k*2*D + d] = 0.0f;
    
                /* Alpha bias to +2 to start with small std dev? Or 0? */
    
                g_b2[k*2*D + (D+d)] = 2.0f; /* Initialize alpha > 0 => exp(-alpha) small? 
    
                                               Wait, u = (u-mu)*exp(-alpha).
    
                                               If alpha=0, exp=1.
    
                                               If alpha=2, exp=0.13.
    
                                               Usually we initialize alpha close to 0.
    
                                            */
    
                g_b2[k*2*D + (D+d)] = 0.0f; 
    
            }
    
    
    
            free(m_in);
    
            free(m_h);
    
        }

    maf_weights_t w;
    w.n_flows = n; w.param_dim = D; w.feature_dim = C; w.hidden_units = H;
    w.M1_data = g_M1; w.M2_data = g_M2; w.perm_data = g_perm; w.inv_perm_data = g_inv_perm;
    w.W1y_data = g_W1y; w.W1c_data = g_W1c; w.b1_data = g_b1;
    w.W2_data = g_W2; w.W2c_data = g_W2c; w.b2_data = g_b2;

    return maf_load_model(&w);
}


/* ========================================================================== */
/* Commands                                                                   */
/* ========================================================================== */

int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

void print_help_train() {
    printf("Usage: maf_cli train [options]\n\n");
    printf("Options:\n");
    printf("  --features <file>    CSV file with feature data (required)\n");
    printf("  --params <file>      CSV file with target parameter data (required)\n");
    printf("  --out <file>         Output model file (default: model.maf)\n");
    printf("  --load <file>        Load initial model weights from file\n");
    printf("  --hidden <int>       Hidden units per layer (default: 16)\n");
    printf("  --blocks <int>       Number of flow blocks (default: 5)\n");
    printf("  --epochs <int>       Training epochs (default: 100)\n");
    printf("  --lr <float>         Learning rate (default: 0.001)\n");
    printf("  --batch <int>        Batch size (default: 32)\n");
    printf("  --skip-header        Skip first line of CSV files\n");
}

void cmd_train(int argc, char** argv) {
    char *feat_file=NULL, *param_file=NULL, *out_file="model.maf", *load_file=NULL;
    int hidden=16, blocks=5, epochs=100, batch_size=32;
    float lr=0.001f;
    bool skip=false;

    for(int i=0; i<argc; i++) {
        if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_help_train();
            exit(0);
        }
        else if(!strcmp(argv[i], "--features")) feat_file = argv[++i];
        else if(!strcmp(argv[i], "--params")) param_file = argv[++i];
        else if(!strcmp(argv[i], "--out")) out_file = argv[++i];
        else if(!strcmp(argv[i], "--load")) load_file = argv[++i];
        else if(!strcmp(argv[i], "--hidden")) hidden = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--blocks")) blocks = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--epochs")) epochs = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--lr")) lr = atof(argv[++i]);
        else if(!strcmp(argv[i], "--batch")) batch_size = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--skip-header")) skip = true;
    }

    if(!feat_file || !param_file) {
        fprintf(stderr, "Error: --features and --params required\n");
        print_help_train();
        exit(1);
    }

    printf("Loading data...\n");
    Dataset* F = load_csv(feat_file, skip);
    Dataset* P = load_csv(param_file, skip);

    if(F->rows != P->rows) {
        fprintf(stderr, "Error: Row mismatch F:%d P:%d\n", F->rows, P->rows);
        exit(1);
    }

    printf("Training: %d samples, C=%d, D=%d\n", F->rows, F->cols, P->cols);
    
    srand(time(NULL));
    maf_model_t* model;
    if (load_file) {
        printf("Loading initial model from %s...\n", load_file);
        model = load_model_file(load_file);
    } else {
        model = init_random_model(blocks, P->cols, F->cols, hidden);
    }
    
    maf_workspace_t* ws = maf_create_workspace(model);
    maf_cache_t* cache = maf_create_cache(model);
    maf_grad_t* grad = maf_create_grad(model);
    maf_adam_t* adam = maf_create_adam(model, lr, 0.9f, 0.999f, 1e-8f);

    /* Training Loop */
    int N = F->rows;
    int C = F->cols;
    int D = P->cols;
    int bs = (batch_size > 0) ? batch_size : N;
    int batches = (N + bs - 1) / bs;

    for(int e=0; e<epochs; e++) {
        float total_loss = 0;
        
        /* Simple sequential batching (shuffle would be better) */
        for(int b=0; b<batches; b++) {
            maf_zero_grad(model, grad);
            float batch_loss = 0;
            int start = b * bs;
            int end = (start + bs > N) ? N : start + bs;
            int count = end - start;

            for(int i=start; i<end; i++) {
                float* f_ptr = &F->data[i*C];
                float* p_ptr = &P->data[i*D];
                
                /* Negative Log Likelihood */
                float logp = maf_forward_train(model, ws, cache, f_ptr, p_ptr);
                batch_loss -= logp;
                
                /* Backward (gradient of NLL) */
                maf_backward(model, cache, grad, f_ptr, p_ptr);
            }
            
            /* Normalize gradients by batch size to mimic Mean Loss */
            /* This makes the optimization behavior independent of batch size */
            if (count > 0) {
                float scale = 1.0f / count;
                for(int k=0; k<model->n_flows; k++) {
                    maf_layer_grad_t* lgrad = &grad->layers[k];
                    int D = model->layers[k].param_dim;
                    int C = model->layers[k].feature_dim;
                    int H = model->layers[k].hidden_units;
                    
                    for(int i=0; i<H*D; i++) lgrad->dW1y[i] *= scale;
                    for(int i=0; i<H*C; i++) lgrad->dW1c[i] *= scale;
                    for(int i=0; i<H; i++) lgrad->db1[i] *= scale;
                    for(int i=0; i<2*D*H; i++) lgrad->dW2[i] *= scale;
                    for(int i=0; i<2*D*C; i++) lgrad->dW2c[i] *= scale;
                    for(int i=0; i<2*D; i++) lgrad->db2[i] *= scale;
                }
            }

            /* Average gradients? Or Sum? 
               Standard SGD assumes mean loss.
               If we sum gradients, we are minimizing Sum Loss.
               Usually we divide by batch size if lr is scaled.
               Here, let's just step.
            */
            maf_adam_step(model, adam, grad);
            total_loss += batch_loss;
        }
        
        if(e % 1000 == 0) printf("Epoch %d: Avg NLL = %f\n", e, total_loss / N);
    }

    save_model(out_file, model);
    
    if(adam) maf_free_adam(adam);
    if(grad) maf_free_grad(grad);
    if(cache) maf_free_cache(cache);
    if(ws) maf_free_workspace(ws);
    if(model) maf_free_model(model);
    free_dataset(F);
    free_dataset(P);
}

void print_help_infer() {
    printf("Usage: maf_cli infer [options]\n\n");
    printf("Options:\n");
    printf("  --model <file>       Path to trained model file (required)\n");
    printf("  --features <file>    CSV file with feature data (required)\n");
    printf("  --out <file>         Output CSV file (default: out.csv)\n");
    printf("  --samples <int>      Number of samples per feature row (default: 1)\n");
    printf("  --mode <string>      Inference mode: 'sample', 'stats', 'quantiles' (default: sample)\n");
    printf("  --quantiles-list <str> Comma-separated list of quantiles (default: 0.05,0.5,0.95)\n");
    printf("  --skip-header        Skip first line of feature CSV\n");
}

void cmd_infer(int argc, char** argv) {
    char *model_file=NULL, *feat_file=NULL, *out_file="out.csv";
    char *mode="sample";
    char *q_list="0.05,0.5,0.95";
    int n_samples=1;
    bool skip=false;

    for(int i=0; i<argc; i++) {
        if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_help_infer();
            exit(0);
        }
        else if(!strcmp(argv[i], "--model")) model_file = argv[++i];
        else if(!strcmp(argv[i], "--features")) feat_file = argv[++i];
        else if(!strcmp(argv[i], "--out")) out_file = argv[++i];
        else if(!strcmp(argv[i], "--samples")) n_samples = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--mode")) mode = argv[++i];
        else if(!strcmp(argv[i], "--quantiles-list")) q_list = argv[++i];
        else if(!strcmp(argv[i], "--skip-header")) skip = true;
    }

    if(!model_file || !feat_file) {
        fprintf(stderr, "Error: --model and --features required\n");
        print_help_infer();
        exit(1);
    }

    maf_model_t* model = load_model_file(model_file);
    Dataset* F = load_csv(feat_file, skip);
    
    FILE* fout = fopen(out_file, "w");
    if(!fout) {
        fprintf(stderr, "Error creating %s\n", out_file);
        exit(1);
    }

    int D = model->param_dim;
    int C = F->cols;
    int N = F->rows;

    /* Parse quantiles */
    float quants[32];
    int n_quants = 0;
    if(!strcmp(mode, "quantiles")) {
        char* qdup = strdup(q_list);
        char* tok = strtok(qdup, ",");
        while(tok) {
            quants[n_quants++] = atof(tok);
            tok = strtok(NULL, ",");
        }
        free(qdup);
        
        /* Header */
        fprintf(fout, "feature_idx,quantile");
        for(int d=0; d<D; d++) fprintf(fout, ",p%d", d);
        fprintf(fout, "\n");
    } else if(!strcmp(mode, "stats")) {
        fprintf(fout, "feature_idx,stat");
        for(int d=0; d<D; d++) fprintf(fout, ",p%d", d);
        fprintf(fout, "\n");
    } else {
        fprintf(fout, "feature_idx,sample_idx");
        for(int d=0; d<D; d++) fprintf(fout, ",p%d", d);
        fprintf(fout, "\n");
    }

    float* samples = malloc(n_samples * D * sizeof(float));

    for(int i=0; i<N; i++) {
        float* f_vec = &F->data[i*C];
        
        /* Generate samples */
        maf_sample(model, f_vec, n_samples, samples, rand());

        if(!strcmp(mode, "sample")) {
            for(int s=0; s<n_samples; s++) {
                fprintf(fout, "%d,%d", i, s);
                for(int d=0; d<D; d++) fprintf(fout, ",%f", samples[s*D+d]);
                fprintf(fout, "\n");
            }
        } 
        else if(!strcmp(mode, "stats")) {
            /* Mean */
            fprintf(fout, "%d,mean", i);
            for(int d=0; d<D; d++) {
                float sum=0;
                for(int s=0; s<n_samples; s++) sum += samples[s*D+d];
                fprintf(fout, ",%f", sum/n_samples);
            }
            fprintf(fout, "\n");
            /* Std */
            fprintf(fout, "%d,std", i);
            for(int d=0; d<D; d++) {
                float sum=0, sum2=0;
                for(int s=0; s<n_samples; s++) {
                    float val = samples[s*D+d];
                    sum += val;
                    sum2 += val*val;
                }
                float mean = sum/n_samples;
                float var = sum2/n_samples - mean*mean;
                fprintf(fout, ",%f", sqrtf(var > 0 ? var : 0));
            }
            fprintf(fout, "\n");
        }
        else if(!strcmp(mode, "quantiles")) {
            /* Sort each dimension independently */
            float* col = malloc(n_samples * sizeof(float));
            for(int d=0; d<D; d++) {
                for(int s=0; s<n_samples; s++) col[s] = samples[s*D+d];
                qsort(col, n_samples, sizeof(float), cmp_float);
                /* Store back sorted (temp hack, efficient enough) */
                for(int s=0; s<n_samples; s++) samples[s*D+d] = col[s];
            }
            free(col);

            for(int q=0; q<n_quants; q++) {
                fprintf(fout, "%d,%g", i, quants[q]);
                int idx = (int)(quants[q] * (n_samples - 1));
                for(int d=0; d<D; d++) {
                    /* samples is row-major [s][d]. We sorted effectively column-wise by extracting? 
                       No, wait. I extracted to 'col', sorted 'col', then put back.
                       But putting back into samples[s*D+d] means samples matrix now has sorted columns.
                    */
                    fprintf(fout, ",%f", samples[idx*D+d]);
                }
                fprintf(fout, "\n");
            }
        }
    }

    free(samples);
    fclose(fout);
    free_dataset(F);
}

void cmd_mcp(int argc, char** argv) {
    char command[4096];
    snprintf(command, sizeof(command), "python3 python/maf_mcp.py");
    for(int i=0; i<argc; i++) {
         strncat(command, " ", sizeof(command) - strlen(command) - 1);
         strncat(command, argv[i], sizeof(command) - strlen(command) - 1);
    }
    int ret = system(command);
    if(ret == -1) {
        fprintf(stderr, "Error executing python/maf_mcp.py\n");
        exit(1);
    }
    /* Simple exit mapping */
    exit(0);
}

void print_help() {
    printf("Usage: maf_cli <command> [options]\n\n");
    printf("Commands:\n");
    printf("  train      Train a MAF model from CSV data\n");
    printf("  infer      Run inference using a trained model\n");
    printf("  mcp        Start the Model Context Protocol (MCP) server\n");
    printf("\nRun 'maf_cli <command> --help' for command-specific help.\n");
}

int main(int argc, char** argv) {
    if(argc < 2) {
        print_help();
        return 1;
    }

    if(!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        print_help();
        return 0;
    }

    if(!strcmp(argv[1], "train")) cmd_train(argc-1, argv+1);
    else if(!strcmp(argv[1], "infer")) cmd_infer(argc-1, argv+1);
    else if(!strcmp(argv[1], "mcp")) cmd_mcp(argc-1, argv+1);
    else {
        printf("Unknown command %s\n", argv[1]);
        print_help();
        return 1;
    }

    return 0;
}
