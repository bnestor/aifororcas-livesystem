#pragma checksum "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "06a73fc98c18a4a3e82af3ceab895264f4fc2916"
// <auto-generated/>
#pragma warning disable 1591
namespace Orcasound.ComponentsLibrary.Map
{
    #line hidden
    using System.Linq;
#line 1 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor"
using System;

#line default
#line hidden
#line 2 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor"
using System.Collections.Generic;

#line default
#line hidden
#line 3 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor"
using System.Threading.Tasks;

#line default
#line hidden
#line 4 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor"
using Microsoft.AspNetCore.Components;

#line default
#line hidden
#line 5 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor"
using Microsoft.JSInterop;

#line default
#line hidden
#line 6 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor"
using Orcasound.ComponentsLibrary.Map;

#line default
#line hidden
    public partial class Map : Microsoft.AspNetCore.Components.ComponentBase
    {
        #pragma warning disable 1998
        protected override void BuildRenderTree(Microsoft.AspNetCore.Components.Rendering.RenderTreeBuilder __builder)
        {
            __builder.OpenElement(0, "div");
            __builder.AddAttribute(1, "id", 
#line 9 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor"
          elementId

#line default
#line hidden
            );
            __builder.AddAttribute(2, "class", "col-lg-12");
            __builder.AddAttribute(3, "style", "height: calc(100vh - 200px); border:1px solid black");
            __builder.CloseElement();
        }
        #pragma warning restore 1998
#line 11 "D:\Repo\OrcaSound\ModeratorFrontEnd\Orcasound\Orcasound.ComponentsLibrary\Map\Map.razor"
       
    string elementId = $"map-{Guid.NewGuid().ToString("D")}";
    
    [Parameter] public double Zoom { get; set; }
    [Parameter] public List<Marker> Markers { get; set; }

    protected async override Task OnAfterRenderAsync(bool firstRender)
    {
        await JSRuntime.InvokeVoidAsync(
            "deliveryMap.showOrUpdate",
            elementId,
            Markers);
    }

#line default
#line hidden
        [global::Microsoft.AspNetCore.Components.InjectAttribute] private IJSRuntime JSRuntime { get; set; }
    }
}
#pragma warning restore 1591
