# Continuous world: SymPDE

#domain = MappedDomain(Cube(), mapping=CollelaMapping())
domain = Domain('Omega')
derham = Derham(domain)

u1, v1 = elements_of(derham.V1, names='u1, v1')
u2, v2 = elements_of(derham.V2, names='u2, v2')

a1 = BilinearForm((u1, v1), integral(domain, dot(u1, v1)))
a2 = BilinearForm((u2, v2), integral(domain, dot(u2, v2)))

#==============================================================================
# Discrete objects: Psydac
domain_h = discretize(domain, ncells=(10, 10, 10))
derham_h = discretize(derham, domain_h, degree=(3, 3, 3), periodic=(False, False, True))

a1_h = discretize(a1, (derham_h.V1, derham_h.V1))
a2_h = discretize(a2, (derham_h.V2, derham_h.V2))

# StencilMatrix objects
M1 = a1_h.assemble()
M2 = a2_h.assemble()

G, C, D = derham_h.derivatives_as_matrices()

# Projectors
#  . Input: callable functions
#  . Output: FemField objects
Pi0, Pi1, Pi2, Pi3 = derham_h.projectors(kind='global', nquads=[10,10,10])
