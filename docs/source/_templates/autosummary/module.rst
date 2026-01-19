{{ fullname | smart_fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :no-index:

   {% block attributes %}
   {% if attributes %}
   Module Attributes
   -----------------

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   Functions
   ---------

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   Classes
   -------
   .. inheritance-diagram:: {{ fullname }}
      :parts: 1
      :top-classes: psydac.linalg.basic.LinearOperator, psydac.linalg.basic.Vector, psydac.linalg.basic.VectorSpace, psydac.linalg.basic.LinearSolver
   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}      
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   Exceptions
   ----------

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
Modules
-------

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

Details
-------

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :exclude-members:
   :show-inheritance:
