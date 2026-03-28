#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "barracuda.h"
#include "bir_cfold.h"
#include "bir_dce.h"
#include "bir_lower.h"
#include "bir_mem2reg.h"
#include "lexer.h"
#include "nvidia.h"
#include "parser.h"
#include "preproc.h"
#include "sema.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

static char source_buf[BC_MAX_SOURCE];
static char pp_out_buf[BC_MAX_SOURCE];
static token_t token_buf[BC_MAX_TOKENS];
static ast_node_t node_buf[BC_MAX_NODES];
static char g_last_error[1024];

static void clear_last_error(void)
{
    g_last_error[0] = '\0';
}

static void set_last_error(const char *message)
{
    if (!message) {
        message = "unknown BarraCUDA error";
    }

    snprintf(g_last_error, sizeof(g_last_error), "%s", message);
}

static void set_phase_error(const char *path, const bc_error_t *error)
{
    if (!error) {
        set_last_error("unknown compiler error");
        return;
    }

    snprintf(g_last_error, sizeof(g_last_error), "%s:%u:%u: E%03u: %s",
             path ? path : "<memory>",
             error->loc.line,
             error->loc.col,
             error->eid,
             error->msg);
}

static int read_file(const char *path, char *buf, uint32_t max, uint32_t *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        set_last_error("failed to open CUDA source file");
        return BC_ERR_IO;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (size < 0 || (uint32_t)size >= max) {
        fclose(fp);
        set_last_error("CUDA source file is too large for the in-browser compiler");
        return BC_ERR_IO;
    }

    *out_len = (uint32_t)fread(buf, 1, (size_t)size, fp);
    buf[*out_len] = '\0';
    fclose(fp);
    return BC_OK;
}

EMSCRIPTEN_KEEPALIVE int barracuda_compile_to_ptx(const char *input_path, const char *output_path)
{
    const char *source_path = input_path ? input_path : "<memory>";
    const char *ptx_path = output_path ? output_path : "/workspace/output.ptx";
    uint32_t source_len = 0;
    int rc = 0;

    clear_last_error();

    if (!input_path || !output_path) {
        set_last_error("compiler paths must not be null");
        return 0;
    }

    if (read_file(input_path, source_buf, BC_MAX_SOURCE, &source_len) != BC_OK) {
        return 0;
    }

    preproc_t *pp = (preproc_t *)malloc(sizeof(preproc_t));
    if (!pp) {
        set_last_error("failed to allocate preprocessor state");
        return 0;
    }

    pp_init(pp, source_buf, source_len, pp_out_buf, BC_MAX_SOURCE, source_path);
    rc = pp_process(pp);
    if (rc != BC_OK || pp->num_errors > 0) {
        if (pp->num_errors > 0) {
            set_phase_error(source_path, &pp->errors[0]);
        } else {
            set_last_error("preprocessing failed");
        }
        free(pp);
        return 0;
    }

    lexer_t lexer;
    lexer_init(&lexer, pp_out_buf, pp->out_len, token_buf, BC_MAX_TOKENS);
    free(pp);

    rc = lexer_tokenize(&lexer);
    if (rc != BC_OK || lexer.num_errors > 0) {
        if (lexer.num_errors > 0) {
            set_phase_error(source_path, &lexer.errors[0]);
        } else {
            set_last_error("lexing failed");
        }
        return 0;
    }

    parser_t parser;
    parser_init(&parser, token_buf, lexer.num_tokens, pp_out_buf, node_buf, BC_MAX_NODES);
    uint32_t root = parser_parse(&parser);
    if (parser.num_errors > 0) {
        set_phase_error(source_path, &parser.errors[0]);
        return 0;
    }

    sema_ctx_t *sema_ctx = (sema_ctx_t *)malloc(sizeof(sema_ctx_t));
    if (!sema_ctx) {
        set_last_error("failed to allocate semantic analysis state");
        return 0;
    }

    sema_init(sema_ctx, &parser, root);
    sema_check(sema_ctx, root);
    if (sema_ctx->num_errors > 0) {
        set_phase_error(source_path, &sema_ctx->errors[0]);
        free(sema_ctx);
        return 0;
    }

    bir_module_t *bir_module = (bir_module_t *)malloc(sizeof(bir_module_t));
    if (!bir_module) {
        set_last_error("failed to allocate BIR module");
        free(sema_ctx);
        return 0;
    }

    bc_error_t lower_errors[BC_MAX_ERRORS];
    int num_lower_errors = 0;
    rc = bir_lower(&parser, root, bir_module, sema_ctx, lower_errors, &num_lower_errors);
    free(sema_ctx);
    if (rc != BC_OK || num_lower_errors > 0) {
        if (num_lower_errors > 0) {
            set_phase_error(source_path, &lower_errors[0]);
        } else {
            set_last_error("IR lowering failed");
        }
        free(bir_module);
        return 0;
    }

    bir_mem2reg(bir_module);
    bir_cfold(bir_module);
    bir_dce(bir_module);

    nv_module_t *nv_module = (nv_module_t *)malloc(sizeof(nv_module_t));
    if (!nv_module) {
        set_last_error("failed to allocate NVIDIA PTX backend state");
        free(bir_module);
        return 0;
    }

    rc = nv_compile(bir_module, nv_module);
    if (rc != BC_OK) {
        set_last_error("BarraCUDA failed to compile the CUDA source to PTX");
        free(nv_module);
        free(bir_module);
        return 0;
    }

    nv_module->bkhit = 0;
    rc = nv_emit_ptx(nv_module, ptx_path);

    free(nv_module);
    free(bir_module);

    if (rc != BC_OK) {
        set_last_error("BarraCUDA generated PTX text but could not write the output file");
        return 0;
    }

    return 1;
}

EMSCRIPTEN_KEEPALIVE const char *barracuda_get_last_error(void)
{
    return g_last_error;
}
