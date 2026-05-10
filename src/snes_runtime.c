#include "snes_runtime.h"
#include "string.h"
#include <immintrin.h>  

typedef struct alloc_hdr {
    u32 size;
    u32 reserved;
} alloc_hdr;

static u8 *g_heap_base;
static u8 *g_heap_ptr;
static u8 *g_heap_end;
static void *g_gadget;
static void *g_sendto_fn;
static s32 g_log_fd = -1;
static u8 g_log_sa[16];

static u32 align16(u32 value) {
    return (value + 15U) & ~15U;
}

static int min_int(int a, int b) {
    return a < b ? a : b;
}

void snes_runtime_init(void *heap_base, u32 heap_size,
                       void *gadget, void *sendto_fn,
                       s32 log_fd, const u8 *log_sa) {
    g_heap_base = (u8 *)heap_base;
    g_heap_ptr = g_heap_base;
    g_heap_end = g_heap_base + heap_size;
    g_gadget = gadget;
    g_sendto_fn = sendto_fn;
    g_log_fd = log_fd;
    for (int i = 0; i < 16; i++) g_log_sa[i] = log_sa ? log_sa[i] : 0;
}

void snes_runtime_reset_heap(void) {
    g_heap_ptr = g_heap_base;
}

void *malloc(size_t size) {
    if (!g_heap_ptr || size == 0) return 0;
    u32 total = align16((u32)size + (u32)sizeof(alloc_hdr));
    if (g_heap_ptr + total > g_heap_end) return 0;
    alloc_hdr *hdr = (alloc_hdr *)g_heap_ptr;
    hdr->size = (u32)size;
    hdr->reserved = 0;
    g_heap_ptr += total;
    return (void *)(hdr + 1);
}

void free(void *ptr) {
    (void)ptr;
}

void *realloc(void *ptr, size_t size) {
    if (!ptr) return malloc(size);
    if (size == 0) {
        free(ptr);
        return 0;
    }
    alloc_hdr *hdr = ((alloc_hdr *)ptr) - 1;
    void *new_ptr = malloc(size);
    if (!new_ptr) return 0;
    memcpy(new_ptr, ptr, (size_t)min_int((int)hdr->size, (int)size));
    return new_ptr;
}

void *memcpy(void *dst, const void *src, size_t size) {
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;

    // AVX2 — 32 bytes per iteration
    // Process as many full 32-byte chunks as possible first
    while (size >= 32) {
        // loadu/storeu handle unaligned pointers safely on x86
        _mm256_storeu_si256((__m256i *)d,
                            _mm256_loadu_si256((const __m256i *)s));
        d += 32; s += 32; size -= 32;
    }
    // 16-byte SSE2 tail
    if (size >= 16) {
        _mm_storeu_si128((__m128i *)d,
                         _mm_loadu_si128((const __m128i *)s));
        d += 16; s += 16; size -= 16;
    }
    // 8-byte tail
    if (size >= 8) {
        *(u64 *)d = *(const u64 *)s;
        d += 8; s += 8; size -= 8;
    }
    // 4-byte tail
    if (size >= 4) {
        *(u32 *)d = *(const u32 *)s;
        d += 4; s += 4; size -= 4;
    }
    // byte tail (at most 3 bytes)
    while (size--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int value, size_t size) {
    u8 *d = (u8 *)dst;
    u8 v = (u8)value;

    // Align to 32-byte boundary first using scalar stores
    // so the AVX2 path always hits aligned addresses
    while (size > 0 && ((u64)d & 31)) {
        *d++ = v;
        size--;
    }

    if (size >= 32) {
        // Broadcast the byte into all 32 lanes of a YMM register
        __m256i wide = _mm256_set1_epi8((char)v);

        // Unrolled 4x = 128 bytes per iteration — reduces loop overhead
        while (size >= 128) {
            _mm256_store_si256((__m256i *)(d +  0), wide);
            _mm256_store_si256((__m256i *)(d + 32), wide);
            _mm256_store_si256((__m256i *)(d + 64), wide);
            _mm256_store_si256((__m256i *)(d + 96), wide);
            d += 128; size -= 128;
        }
        // remaining 32-byte chunks
        while (size >= 32) {
            _mm256_store_si256((__m256i *)d, wide);
            d += 32; size -= 32;
        }
        // 16-byte tail
        if (size >= 16) {
            _mm_store_si128((__m128i *)d, _mm256_castsi256_si128(wide));
            d += 16; size -= 16;
        }
    }

    // byte tail
    while (size--) *d++ = v;
    return dst;
}

void *memmove(void *dst, const void *src, size_t size) {
    u8 *d = (u8 *)dst;
    const u8 *s = (const u8 *)src;

    if (d == s || size == 0) return dst;

    if (d < s || d >= s + size) {
        // No overlap OR dst is fully after src — safe forward copy
        // Identical to memcpy AVX2 path
        while (size >= 32) {
            _mm256_storeu_si256((__m256i *)d,
                                _mm256_loadu_si256((const __m256i *)s));
            d += 32; s += 32; size -= 32;
        }
        if (size >= 16) {
            _mm_storeu_si128((__m128i *)d,
                             _mm_loadu_si128((const __m128i *)s));
            d += 16; s += 16; size -= 16;
        }
        if (size >= 8) {
            *(u64 *)d = *(const u64 *)s; d += 8; s += 8; size -= 8;
        }
        if (size >= 4) {
            *(u32 *)d = *(const u32 *)s; d += 4; s += 4; size -= 4;
        }
        while (size--) *d++ = *s++;

    } else {
        // Overlapping and dst > src — must copy backwards
        // Advance to end of both buffers
        d += size;
        s += size;

        // Byte-align the tail (from the end) before switching to AVX2
        while (size > 0 && ((u64)d & 31)) {
            *--d = *--s;
            size--;
        }

        // AVX2 backward — 32 bytes per iteration
        while (size >= 32) {
            d -= 32; s -= 32;
            _mm256_storeu_si256((__m256i *)d,
                                _mm256_loadu_si256((const __m256i *)s));
            size -= 32;
        }
        // 16-byte backward tail
        if (size >= 16) {
            d -= 16; s -= 16;
            _mm_storeu_si128((__m128i *)d,
                             _mm_loadu_si128((const __m128i *)s));
            size -= 16;
        }
        // 8-byte backward tail
        if (size >= 8) {
            d -= 8; s -= 8;
            *(u64 *)d = *(const u64 *)s;
            size -= 8;
        }
        // byte tail
        while (size--) *--d = *--s;
    }

    return dst;
}

int memcmp(const void *a, const void *b, size_t size) {
    const u8 *aa = (const u8 *)a;
    const u8 *bb = (const u8 *)b;
    for (size_t i = 0; i < size; i++) {
        if (aa[i] != bb[i]) return (int)aa[i] - (int)bb[i];
    }
    return 0;
}

void snes_runtime_log(const char *msg) {
    if (!msg || g_log_fd < 0 || !g_sendto_fn) return;
    int len = 0;
    while (msg[len]) len++;
    NC(g_gadget, g_sendto_fn, (u64)g_log_fd, (u64)msg, (u64)len, 0, (u64)g_log_sa, 16);
}

static int append_char(char *buf, int pos, int max, char ch) {
    if (pos < max - 1) buf[pos] = ch;
    return pos + 1;
}

static int append_str(char *buf, int pos, int max, const char *str) {
    const char *s = str ? str : "(null)";
    while (*s) {
        pos = append_char(buf, pos, max, *s);
        s++;
    }
    return pos;
}

static int append_u32_base(char *buf, int pos, int max, u32 value, u32 base, int upper) {
    char tmp[16];
    int n = 0;
    if (value == 0) return append_char(buf, pos, max, '0');
    while (value && n < (int)sizeof(tmp)) {
        u32 digit = value % base;
        tmp[n++] = (char)(digit < 10 ? ('0' + digit) : ((upper ? 'A' : 'a') + (digit - 10)));
        value /= base;
    }
    while (n > 0) pos = append_char(buf, pos, max, tmp[--n]);
    return pos;
}

static int append_s32(char *buf, int pos, int max, s32 value) {
    if (value < 0) {
        pos = append_char(buf, pos, max, '-');
        return append_u32_base(buf, pos, max, (u32)(-value), 10, 0);
    }
    return append_u32_base(buf, pos, max, (u32)value, 10, 0);
}

void snes_runtime_vlogf(const char *fmt, va_list ap) {
    char buf[512];
    int pos = 0;
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') {
            pos = append_char(buf, pos, (int)sizeof(buf), *p);
            continue;
        }
        p++;
        if (!*p) break;
        switch (*p) {
        case '%':
            pos = append_char(buf, pos, (int)sizeof(buf), '%');
            break;
        case 'c':
            pos = append_char(buf, pos, (int)sizeof(buf), (char)va_arg(ap, int));
            break;
        case 's':
            pos = append_str(buf, pos, (int)sizeof(buf), va_arg(ap, const char *));
            break;
        case 'd':
        case 'i':
            pos = append_s32(buf, pos, (int)sizeof(buf), va_arg(ap, int));
            break;
        case 'u':
            pos = append_u32_base(buf, pos, (int)sizeof(buf), va_arg(ap, unsigned int), 10, 0);
            break;
        case 'x':
            pos = append_u32_base(buf, pos, (int)sizeof(buf), va_arg(ap, unsigned int), 16, 0);
            break;
        case 'X':
            pos = append_u32_base(buf, pos, (int)sizeof(buf), va_arg(ap, unsigned int), 16, 1);
            break;
        default:
            pos = append_char(buf, pos, (int)sizeof(buf), '%');
            pos = append_char(buf, pos, (int)sizeof(buf), *p);
            break;
        }
    }
    if (pos >= (int)sizeof(buf)) pos = (int)sizeof(buf) - 1;
    buf[pos] = 0;
    snes_runtime_log(buf);
}

void snes_runtime_logf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    snes_runtime_vlogf(fmt, ap);
    va_end(ap);
}

int vprintf(const char *fmt, va_list ap) {
    va_list copy;
    va_copy(copy, ap);
    snes_runtime_vlogf(fmt, copy);
    va_end(copy);
    return 0;
}

int printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    snes_runtime_vlogf(fmt, ap);
    va_end(ap);
    return 0;
}
