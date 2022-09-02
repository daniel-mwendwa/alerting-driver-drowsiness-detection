import subprocess
import re
command_output = subprocess.run(["netsh", "wlan", "show", "profile"], capture_output=True).stdout.decode()
profile_names = (re.findall("all user profile.....:.(.*)\r", command_output))
wifi_list = list()
if len(profile_names) != 0:
    for name in profile_names:
        wifi_profile = dict()
        profile_info = subprocess.run(["netsh", "wlan", "show", "profile", name], capture_output=True).stdout.decode()
        if re.search("security key:Absent", profile_info):
            continue
        else:
            wifi_profile["ssid"] = name
            profile_info_pass = subprocess.run(["netsh", "wlan", "show", "profile", name, "key=clear"])
            password = re.search("key content(.*)\r", profile_info_pass)
            if password is None:
                wifi_profile["password"] = None
            else:
                wifi_profile["password"] = password[1]
            wifi_list.append(wifi_profile)
for x in range(len(wifi_list)):
    print(wifi_list[x])
