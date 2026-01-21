# Retry with SymPy; produce explicit string output if pretty printing fails.
import sympy as sp

# Define symbols
M1, M2, C, K, a = sp.symbols('M1 M2 C K a', real=True)
w2 = sp.symbols('w2', real=True)  # use w2 to avoid unicode issues

# Quadratic in x = ω^2 (here 'w2')
x = sp.symbols('x', real=True)
theta = K*a
eq_x = sp.Eq(M1*M2*x**2 - 2*C*(M1+M2)*x + 2*C**2*(1 - sp.cos(theta)), 0)

# Solve
sol = sp.solve(eq_x, x)

# Simplify to compact symmetric form
sol_compact = [
    sp.simplify(C/(M1*M2) * ((M1+M2) + sp.sqrt(M1**2 + M2**2 + 2*M1*M2*sp.cos(theta)))),
    sp.simplify(C/(M1*M2) * ((M1+M2) - sp.sqrt(M1**2 + M2**2 + 2*M1*M2*sp.cos(theta))))
]

# Display
print("Equation in x = ω²:")
print(eq_x)
print("\nSolutions for ω² (raw from solve):")
for s in sol:
    print("ω² =", s)

print("\nSolutions for ω² (compact symmetric form):")
for s in sol_compact:
    print("ω² =", s)

# Special case M1 = M2 = M
M = sp.symbols('M', positive=True, real=True)
special = [sp.simplify(s.subs({M1:M, M2:M})) for s in sol_compact]

print("\nSpecial case M1 = M2 = M:")
for s in special:
    print("ω² =", s)

# Half-angle rewrite showing Abs(cos(Ka/2)) for general correctness
theta = K*a
half_angle_plus  = sp.simplify((2*C/M)*(1 + sp.Abs(sp.cos(theta/2))))
half_angle_minus = sp.simplify((2*C/M)*(1 - sp.Abs(sp.cos(theta/2))))

print("\nSpecial case forms using half-angle (SymPy uses Abs for general θ):")
print("ω²_+ =", half_angle_plus)
print("ω²_- =", half_angle_minus)

# If we add a local assumption cos(Ka/2) >= 0 (e.g., near Γ), show familiar forms:
opt = sp.simplify((2*C/M)*(1 + sp.cos(theta/2)))
ac  = sp.simplify((2*C/M)*(1 - sp.cos(theta/2)))
print("\nAssuming cos(Ka/2) ≥ 0 (near Γ), the forms reduce to:")
print("ω²_optical =", opt)
print("ω²_acoustic =", ac)
