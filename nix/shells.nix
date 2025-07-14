{ inputs, ... }:
{
  perSystem =
    {
      config,
      pkgs,
      system,
      ...
    }:
    {
      devShells = {
        default =
          let
            name = "dafny2verus";
            shellHook = "echo ${name}";
            buildInputs = import ./buildInputs.nix {
              inherit (inputs) nixpkgs;
              inherit system;
            };
          in
          pkgs.mkShell { inherit name shellHook buildInputs; };
        spexus = config.nci.outputs.spexus.devShell;
      };
    };
}
