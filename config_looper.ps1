Write-Host "Collecting Config_files"
Get-ChildItem "." -Filter config*.yml
Get-ChildItem "." -Filter config*.yml | Foreach-Object {
    $fileName = $_.Name
    $fileName   # Optional for returning the file name to the console
    python -m scsavailability.run --config=$fileName
}