{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

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
.. rubric:: Modules

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

