.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)
.. _abstract:

Abstract SPL object
===================

All objects in **SPL** are extensions of the abstract class **spl_t_abstract**. This allows us to *propagate* different features to all our objects. For instance, the subroutine **free** is a *deferred* method of **spl_t_abstract**. Hence, all **SPL** objects must implement it.

Deferred methods
****************

free
^^^^

Here's the *abstract interface* of the **free** method

.. code-block:: fortran
   
  ! ..................................................
  abstract interface
     subroutine spl_p_free_abstract(self)
       import spl_t_abstract

       class(spl_t_abstract), intent(inout)  :: self
     end subroutine spl_p_free_abstract
  end interface
  ! ..................................................

Attributs
*********

is_allocated
^^^^^^^^^^^^

is a *logical* attribut. When the concrete object allocates all its memory, this flag is set to **True**. Otherwise, it is **False**.

.. Local Variables:
.. mode: rst
.. End:
