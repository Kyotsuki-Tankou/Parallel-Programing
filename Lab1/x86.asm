;x86
mainMul(int):
        push    rbp
        mov     rbp, rsp
        sub     rsp, 40
        mov     DWORD PTR [rbp-36], edi
        mov     QWORD PTR [rbp-8], 0
        mov     eax, DWORD PTR [rbp-36]
        cdqe
        mov     rdi, rax
        call    init(unsigned long long)
        mov     DWORD PTR [rbp-12], 0
.L16:
        mov     eax, DWORD PTR [rbp-12]
        cmp     eax, DWORD PTR [rbp-36]
        jge     .L13
        mov     DWORD PTR [rbp-16], 0
.L15:
        mov     eax, DWORD PTR [rbp-16]
        cmp     eax, DWORD PTR [rbp-36]
        jge     .L14
        mov     eax, DWORD PTR [rbp-12]
        cdqe
        mov     rcx, QWORD PTR sum[0+rax*8]
        mov     eax, DWORD PTR [rbp-16]
        cdqe
        mov     rdx, QWORD PTR a[0+rax*8]
        mov     eax, DWORD PTR [rbp-12]
        movsx   rsi, eax
        mov     eax, DWORD PTR [rbp-16]
        cdqe
        imul    rax, rax, 11451
        add     rax, rsi
        mov     rax, QWORD PTR b[0+rax*8]
        imul    rax, rdx
        lea     rdx, [rcx+rax]
        mov     eax, DWORD PTR [rbp-12]
        cdqe
        mov     QWORD PTR sum[0+rax*8], rdx
        add     DWORD PTR [rbp-16], 1
        jmp     .L15
.L14:
        add     DWORD PTR [rbp-12], 1
        jmp     .L16
.L13:
        mov     DWORD PTR [rbp-20], 0
.L18:
        mov     eax, DWORD PTR [rbp-20]
        cmp     eax, DWORD PTR [rbp-36]
        jge     .L17
        mov     eax, DWORD PTR [rbp-20]
        cdqe
        mov     rax, QWORD PTR sum[0+rax*8]
        add     QWORD PTR [rbp-8], rax
        add     DWORD PTR [rbp-20], 1
        jmp     .L18
.L17:
        mov     rax, QWORD PTR [rbp-8]
        leave
        ret
.LC0:
        .string ","