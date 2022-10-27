.. role:: hidden
    :class: hidden-section

mmeval.fileio
===================================

.. contents:: mmeval.fileio
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmeval.fileio

File Backend
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseStorageBackend
   LocalBackend
   HTTPBackend
   LmdbBackend
   MemcachedBackend
   PetrelBackend

.. autosummary::
   :toctree: generated
   :nosignatures:

   register_backend

File Handler
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseFileHandler
   JsonHandler
   PickleHandler
   YamlHandler

.. autosummary::
   :toctree: generated
   :nosignatures:

   register_handler

File IO
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load
   exists
   get
   get_file_backend
   get_local_path
   get_text
   isdir
   isfile
   join_path
   list_dir_or_file

Parse File
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dict_from_file
   list_from_file
