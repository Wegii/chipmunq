# shell.nix
with import <nixpkgs> {};

mkShell rec {
  buildInputs = [
    python312
    python312Packages.virtualenv
    cmake
    graphviz
    fontconfig
    dejavu_fonts

    rustc
    cargo
  ];

  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc
    zlib
  ];
  LD_LIBRARY_PATH = NIX_LD_LIBRARY_PATH;
  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";

  FONTCONFIG_FILE = "${fontconfig.out}/etc/fonts/fonts.conf";
  FONTCONFIG_PATH = "${fontconfig.out}/etc/fonts";

  shellHook = ''
    # Update font cache so Graphviz can find fonts
    fc-cache -f ${dejavu_fonts}/share/fonts

    # Activate virtualenv if it exists
    #if [ -d ".venv" ]; then
    #source ../venv/qeccm_new/bin/activate
    #else
    #  echo "No virtual environment found, create it using python -m venv .venv"
    #fi
  '';
}
