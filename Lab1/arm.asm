;arm
mainMul(int):                            // @mainMul(int)
        sub     sp, sp, #48
        stp     x29, x30, [sp, #32]             // 16-byte Folded Spill
        add     x29, sp, #32
        stur    w0, [x29, #-4]
        str     xzr, [sp, #16]
        ldursw  x0, [x29, #-4]
        bl      init(unsigned long long)
        str     wzr, [sp, #12]
        b       .LBB1_1
.LBB1_1:                                // =>This Loop Header: Depth=1
        ldr     w8, [sp, #12]
        ldur    w9, [x29, #-4]
        subs    w8, w8, w9
        b.ge    .LBB1_8
        b       .LBB1_2
.LBB1_2:                                //   in Loop: Header=BB1_1 Depth=1
        str     wzr, [sp, #8]
        b       .LBB1_3
.LBB1_3:                                //   Parent Loop BB1_1 Depth=1
        ldr     w8, [sp, #8]
        ldur    w9, [x29, #-4]
        subs    w8, w8, w9
        b.ge    .LBB1_6
        b       .LBB1_4
.LBB1_4:                                //   in Loop: Header=BB1_3 Depth=2
        ldrsw   x9, [sp, #8]
        adrp    x8, a
        add     x8, x8, :lo12:a
        ldr     x8, [x8, x9, lsl #3]
        ldrsw   x9, [sp, #8]
        mov     x10, #26072                     // =0x65d8
        movk    x10, #1, lsl #16
        mul     x10, x9, x10
        adrp    x9, b
        add     x9, x9, :lo12:b
        add     x9, x9, x10
        ldrsw   x10, [sp, #12]
        ldr     x9, [x9, x10, lsl #3]
        mul     x10, x8, x9
        ldrsw   x9, [sp, #12]
        adrp    x8, sum
        add     x8, x8, :lo12:sum
        add     x9, x8, x9, lsl #3
        ldr     x8, [x9]
        add     x8, x8, x10
        str     x8, [x9]
        b       .LBB1_5
.LBB1_5:                                //   in Loop: Header=BB1_3 Depth=2
        ldr     w8, [sp, #8]
        add     w8, w8, #1
        str     w8, [sp, #8]
        b       .LBB1_3
.LBB1_6:                                //   in Loop: Header=BB1_1 Depth=1
        b       .LBB1_7
.LBB1_7:                                //   in Loop: Header=BB1_1 Depth=1
        ldr     w8, [sp, #12]
        add     w8, w8, #1
        str     w8, [sp, #12]
        b       .LBB1_1
.LBB1_8:
        str     wzr, [sp, #4]
        b       .LBB1_9
.LBB1_9:                                // =>This Inner Loop Header: Depth=1
        ldr     w8, [sp, #4]
        ldur    w9, [x29, #-4]
        subs    w8, w8, w9
        b.ge    .LBB1_12
        b       .LBB1_10
.LBB1_10:                               //   in Loop: Header=BB1_9 Depth=1
        ldrsw   x9, [sp, #4]
        adrp    x8, sum
        add     x8, x8, :lo12:sum
        ldr     x9, [x8, x9, lsl #3]
        ldr     x8, [sp, #16]
        add     x8, x8, x9
        str     x8, [sp, #16]
        b       .LBB1_11
.LBB1_11:                               //   in Loop: Header=BB1_9 Depth=1
        ldr     w8, [sp, #4]
        add     w8, w8, #1
        str     w8, [sp, #4]
        b       .LBB1_9
.LBB1_12:
        ldr     x0, [sp, #16]
        ldp     x29, x30, [sp, #32]             // 16-byte Folded Reload
        add     sp, sp, #48
        ret