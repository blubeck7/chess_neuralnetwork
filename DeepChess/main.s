	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.intel_syntax noprefix
	.globl	_main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## BB#0:
	push	rbp
Lcfi0:
	.cfi_def_cfa_offset 16
Lcfi1:
	.cfi_offset rbp, -16
	mov	rbp, rsp
Lcfi2:
	.cfi_def_cfa_register rbp
	mov	dword ptr [rbp - 4], 0
	mov	dword ptr [rbp - 8], edi
	mov	qword ptr [rbp - 16], rsi
	mov	dword ptr [rbp - 20], 2
	mov	dword ptr [rbp - 24], 3
	mov	edi, dword ptr [rbp - 20]
	add	edi, dword ptr [rbp - 24]
	mov	dword ptr [rbp - 28], edi
	mov	eax, dword ptr [rbp - 28]
	pop	rbp
	ret
	.cfi_endproc


.subsections_via_symbols
