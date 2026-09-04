import { DepthAcceptanceBars } from "./DepthAcceptanceBars";
import { VerifyWaterfall } from "./VerifyWaterfall";
import {
  CorrectionBonusTile,
  ServerTokSTile,
  VerifyRatioTile,
} from "./VerifyTiles";

export function SpeculativeTab() {
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-6">
        <DepthAcceptanceBars />
      </div>
      <div className="col-span-12 lg:col-span-6">
        <VerifyWaterfall />
      </div>

      <div className="col-span-12 lg:col-span-4">
        <VerifyRatioTile />
      </div>
      <div className="col-span-12 lg:col-span-4">
        <CorrectionBonusTile />
      </div>
      <div className="col-span-12 lg:col-span-4">
        <ServerTokSTile />
      </div>
    </div>
  );
}
