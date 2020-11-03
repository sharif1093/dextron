import pkgutil
package = pkgutil.get_loader("dextron")
print(">> <dextron> is being loaded from:", package.path)
