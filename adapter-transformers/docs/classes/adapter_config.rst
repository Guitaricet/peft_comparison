Adapter Configuration
=======================

Classes representing the architectures of adapter modules and fusion layers.

Single (bottleneck) adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.AdapterConfigBase
    :members:

.. autoclass:: adapters.BnConfig
    :members:
    :inherited-members: Mapping

.. autoclass:: adapters.SeqBnConfig
    :members:

.. autoclass:: adapters.SeqBnInvConfig
    :members:

.. autoclass:: adapters.DoubleSeqBnConfig
    :members:

.. autoclass:: adapters.DoubleSeqBnInvConfig
    :members:

.. autoclass:: adapters.ParBnConfig
    :members:

.. autoclass:: adapters.CompacterConfig
    :members:

.. autoclass:: adapters.CompacterPlusPlusConfig
    :members:

Prefix Tuning
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.PrefixTuningConfig
    :members:
    :inherited-members: Mapping

LoRAConfig
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.LoRAConfig
    :members:
    :inherited-members: Mapping

IA3Config
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.IA3Config
    :members:
    :inherited-members: Mapping

Combined configurations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.ConfigUnion
    :members:
    :inherited-members: Mapping

.. autoclass:: adapters.MAMConfig
    :members:

.. autoclass:: adapters.UniPELTConfig
    :members:

Adapter Fusion
~~~~~~~~~~~~~~~

.. autoclass:: adapters.AdapterFusionConfig
    :members:
    :inherited-members: Mapping

.. autoclass:: adapters.StaticAdapterFusionConfig
    :members:

.. autoclass:: adapters.DynamicAdapterFusionConfig
    :members:

Adapter Setup
~~~~~~~~~~~~~~~

.. autoclass:: adapters.AdapterSetup
    :members:
