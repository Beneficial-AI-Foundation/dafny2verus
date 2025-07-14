{ nixpkgs, system }:
let
  pkgs = import nixpkgs {
    config.allowUnfree = true;
    inherit system;
  };
in
with pkgs;
[
  uv
  lefthook
  claude-code
  dafny
  cargo
  rustup
]
