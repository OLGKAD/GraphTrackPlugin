//
//  main.hpp
//  GraphTrackPlugin
//
//  Created by Olzhas Kadyrakun on 26/12/2018.
//  Copyright © 2018 Olzhas Kadyrakunov. All rights reserved.
//

#pragma once
#if UNITY_METRO
#define EXPORT_API __declspec(dllexport) __stdcall
#elif UNITY_WIN
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif
