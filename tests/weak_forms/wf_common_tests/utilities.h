
#include <deal.II/base/exceptions.h>


DeclException4(ExcMatrixEntriesNotEqual,
               int,
               int,
               double,
               double,
               << "Matrix entries are different (exemplar). "
               << "(R,C) = (" << arg1 << "," << arg2 << "). "
               << "Blessed value: " << arg3 << "; "
               << "Other value: " << arg4 << ".");

DeclException2(ExcIteratorRowIndexNotEqual,
               int,
               int,
               << "Iterator row index mismatch. "
               << "  Iterator 1: " << arg1 << "  Iterator 2: " << arg2);

DeclException2(ExcIteratorColumnIndexNotEqual,
               int,
               int,
               << "Iterator column index mismatch. "
               << "  Iterator 1: " << arg1 << "  Iterator 2: " << arg2);


DeclException3(ExcVectorEntriesNotEqual,
               int,
               double,
               double,
               << "Vector entries are different (exemplar). "
               << "(R) = (" << arg1 << "). "
               << "Blessed value: " << arg2 << "; "
               << "Other value: " << arg3 << ".");