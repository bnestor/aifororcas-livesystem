#pragma checksum "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\Pages\FetchData.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "f0504128e11934b2394ec2dfa11dc43a2cb5aa38"
// <auto-generated/>
#pragma warning disable 1591
#pragma warning disable 0414
#pragma warning disable 0649
#pragma warning disable 0169

namespace Orcasound.UI.Pages
{
    #line hidden
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Microsoft.AspNetCore.Components;
#nullable restore
#line 1 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using System.Net.Http;

#line default
#line hidden
#nullable disable
#nullable restore
#line 2 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Microsoft.AspNetCore.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Microsoft.AspNetCore.Components.Authorization;

#line default
#line hidden
#nullable disable
#nullable restore
#line 4 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Microsoft.AspNetCore.Components.Forms;

#line default
#line hidden
#nullable disable
#nullable restore
#line 5 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Microsoft.AspNetCore.Components.Routing;

#line default
#line hidden
#nullable disable
#nullable restore
#line 6 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Microsoft.AspNetCore.Components.Web;

#line default
#line hidden
#nullable disable
#nullable restore
#line 7 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#nullable disable
#nullable restore
#line 8 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Orcasound.UI;

#line default
#line hidden
#nullable disable
#nullable restore
#line 9 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Orcasound.UI.Shared;

#line default
#line hidden
#nullable disable
#nullable restore
#line 10 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Orcasound.Shared.Entities;

#line default
#line hidden
#nullable disable
#nullable restore
#line 11 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Orcasound.UI.Services;

#line default
#line hidden
#nullable disable
#nullable restore
#line 13 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Orcasound.ComponentsLibrary;

#line default
#line hidden
#nullable disable
#nullable restore
#line 14 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\_Imports.razor"
using Orcasound.ComponentsLibrary.Map;

#line default
#line hidden
#nullable disable
#nullable restore
#line 3 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\Pages\FetchData.razor"
using Orcasound.UI.Data;

#line default
#line hidden
#nullable disable
    [Microsoft.AspNetCore.Components.RouteAttribute("/fetchdata")]
    public partial class FetchData : Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
        }
        #pragma warning restore 1998
#nullable restore
#line 39 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.UI\Pages\FetchData.razor"
       
    private WeatherForecast[] forecasts;

    protected override async Task OnInitializedAsync()
    {
        forecasts = await ForecastService.GetForecastAsync(DateTime.Now);
    }

#line default
#line hidden
#nullable disable
        [global::Microsoft.AspNetCore.Components.InjectAttribute] private WeatherForecastService ForecastService { get; set; }
    }
}
#pragma warning restore 1591